

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import argparse
from PIL import Image
import models_mt
import util.lr_sched as lr_sched
import re
from torch.utils.tensorboard import SummaryWriter
import logging
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import warnings
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload

warnings.filterwarnings('ignore', category=UserWarning)

def set_seed(seed=42):
    """设置全局随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

class AutomaticWeightedLoss(nn.Module):
    """自动任务权重损失模块"""
    def __init__(self, num_tasks):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num_tasks, requires_grad=True))
        
    def forward(self, losses):
        """计算加权损失"""
        eps = 1e-8
        total_loss = 0.0
        for i, loss in enumerate(losses):
            total_loss += 0.5 / (self.params[i]**2 + eps) * loss + torch.log(1 + self.params[i]**2)
        return total_loss

class MultiTaskDataset(Dataset):
    def __init__(self, data_root, img_types, transform=None, nb_classes=None):
        self.transform = transform
        self.image_dir = os.path.join(data_root, 'images')
        self.label_root = os.path.join(data_root, 'labels')
        self.img_types = img_types
        self.nb_classes = nb_classes

        # 1. 获取所有有效图像文件（确保所有任务标签都存在）
        all_files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg'))
        ]
        
        self.image_files = []
        for f in all_files:
            prefix = os.path.splitext(f)[0]
            if all(
                os.path.exists(os.path.join(self.label_root, task, f"{prefix}.npy"))
                for task in self.img_types
            ):
                self.image_files.append(f)

        if len(self.image_files) == 0:
            raise RuntimeError("No valid samples found. Please check label completeness.")

        self.file_prefixes = [os.path.splitext(f)[0] for f in self.image_files]

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        prefix = self.file_prefixes[idx]

        if self.transform:
            image = self.transform(image)

        labels = {}
        for task in self.img_types:
            label_path = os.path.join(self.label_root, task, f"{prefix}.npy")
            label_data = np.load(label_path)
            label = torch.from_numpy(label_data.astype(np.int64)).squeeze()
            assert (label >= 0).all() and (label < self.nb_classes).all(), \
                f"Invalid label: {label} for task: {task} at {label_path}"
            labels[task] = label

        return {'image': image, 'filename': prefix, **labels}

    def __len__(self):
        return len(self.image_files)

def custom_collate_fn(batch):
    collated = {'image': torch.stack([i['image'] for i in batch])}
    for key in batch[0]:
        if key not in ['image', 'filename']:
            collated[key] = torch.stack([i[key] for i in batch])
    return collated

def calculate_auc(preds, targets, task):
    try:
        if len(np.unique(targets)) > 1:
            auc = roc_auc_score(targets, preds)
            return auc if not np.isnan(auc) else 0.5
        return 0.5
    except ValueError:
        return 0.5

def init_distributed_mode(args):
    """初始化分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.local_rank = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

def main(args):

    if not hasattr(args, 'rank'):
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        args.distributed = False

    if torch.cuda.device_count() > 1:
        init_distributed_mode(args)  
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.distributed = False

        args.rank = getattr(args, 'rank', 0)
        args.world_size = getattr(args, 'world_size', 1)
        args.local_rank = getattr(args, 'local_rank', 0)
    
    device = torch.device(args.device)
    if args.distributed:
        device = torch.device('cuda', args.local_rank)
    

    if args.rank == 0:  # 仅主进程创建目录
        save_dir = os.path.join("models", 'my_models') ###########################################################模型保存地址

        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(args.dataset, 'logs_0801'), exist_ok=True)
    
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')

    # ===== 3. 数据预处理优化 =====
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    # ===== 4. 数据加载优化 =====
    full_dataset = MultiTaskDataset(
        data_root=args.dataset,
        img_types=args.img_types,
        transform=None,
        nb_classes=args.nb_classes
    )
   
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # 应用transform
    train_subset.dataset.transform = train_transform
    val_subset.dataset.transform = val_transform

    # ===== 分布式采样器 =====
    if args.distributed:
        train_sampler = DistributedSampler(train_subset, shuffle=True)
        val_sampler = DistributedSampler(val_subset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=custom_collate_fn
    )

    
    model = models_mt.__dict__[args.model](
        img_types=args.img_types,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    ).to(device)
    
   
    if args.distributed:
        model = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            cpu_offload=None,                 
        )
        print(f"Rank {args.rank}: 使用 {args.world_size} 个GPU并行训练 (FSDP-ZeRO3)")
    elif torch.cuda.device_count() > 1:
        # 单节点多卡但不想用 FSDP 时保留 DataParallel
        model = nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 个GPU并行训练 (DataParallel模式)")
    
    # 启用梯度检查点
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    criterion = {t: nn.CrossEntropyLoss() for t in args.img_types}
    awl = AutomaticWeightedLoss(len(args.img_types)).to(device)
    
    # 优化器配置 - 学习率按GPU数量线性放大[6](@ref)
    base_lr = args.lr * args.world_size if args.distributed else args.lr
    optimizer = optim.SGD([
        {'params': model.parameters(), 'weight_decay': 1e-4},
        {'params': awl.parameters(), 'lr': base_lr * 0.1}
    ], lr=base_lr, momentum=0.9)

    # ===== 6. 混合精度训练核心配置 =====
    scaler = torch.cuda.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ===== 7. 日志配置 =====
    if args.rank == 0:  # 仅主进程初始化日志
        writer = SummaryWriter(log_dir=os.path.join(args.dataset, 'logs'))
        logging.basicConfig(
            filename=os.path.join(args.dataset, 'logs_0801', f'log_rank{args.rank}.txt'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        logger = logging.getLogger()

    # ===== 8. 训练循环优化 =====
    best_val_auc = 0.0
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()
        train_metrics = defaultdict(float)
        train_preds = {t: [] for t in args.img_types}
        train_targets = {t: [] for t in args.img_types}
        
        optimizer.zero_grad(set_to_none=True)

        # 训练阶段
        with tqdm(train_loader, desc=f"Rank {args.rank} Epoch {epoch+1}/{args.epochs} Train", 
                  disable=not (args.rank == 0)) as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(device, non_blocking=True)
                labels = {t: batch[t].to(device, non_blocking=True) for t in args.img_types}
                
                # GPU归一化
                images = transforms.functional.normalize(
                    images, 
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
                
                # ===== 混合精度计算核心 =====
                with torch.cuda.amp.autocast():
                    outputs, aux_loss = model(images, task=None)
                    task_losses = []
                    for task in args.img_types:
                        task_loss = criterion[task](outputs[task], labels[task])
                        task_losses.append(task_loss)

                    total_loss = awl(task_losses) + aux_loss

                # ===== 梯度累积与更新 =====
                scaler.scale(total_loss).backward()

                if (batch_idx + 1) % args.accum_iter == 0:
                    # 把可能落在 CPU 的梯度拉回 CUDA，避免 AMP unscale 报错
                    # for group in optimizer.param_groups:
                    #     for p in group['params']:
                    #         if p.grad is not None and p.grad.device.type != 'cuda':
                    #             p.grad = p.grad.to('cuda')
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                # ===== 指标记录 =====
                with torch.no_grad():
                    for task in args.img_types:
                        probs = torch.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()
                        train_preds[task].extend(probs)
                        train_targets[task].extend(labels[task].cpu().numpy())
                
                # 计算并累加每个任务的准确率
                for t in args.img_types:
                    acc = (outputs[t].argmax(1) == labels[t]).float().mean().item()
                    train_metrics[f"{t}_acc"] += acc
                    train_metrics[f"{t}_loss"] += task_losses[args.img_types.index(t)].item()
                
                train_metrics["total_loss"] += total_loss.item()
                
                pbar.set_postfix({
                    "Loss": f"{train_metrics['total_loss'] / (batch_idx + 1):.4f}",
                    **{f"{t} Acc": f"{train_metrics[f'{t}_acc'] / (batch_idx + 1):.4f}"
                    for t in args.img_types}
                })

                # 专家内存整理
                if hasattr(model, 'module'):
                    model_blocks = model.module.blocks
                else:
                    model_blocks = model.blocks
                    
                for blk in model_blocks:
                    if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'step_counter'):
                        blk.mlp.step_counter += 1
                        if blk.mlp.step_counter % 25 == 0:
                            blk.mlp._defragment_memory()

                    if hasattr(blk, 'attn') and hasattr(blk.attn, 'step_counter'):
                        blk.attn.step_counter += 1
                        if blk.attn.step_counter % 25 == 0:
                            blk.attn._defragment_memory()

                # 显存清理
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        
        # ===== 学习率调整 =====
        scheduler.step()
        
        # ===== 训练指标计算 =====
        train_auc = {task: calculate_auc(train_preds[task], train_targets[task], task)
                     for task in args.img_types}
        
        # ===== 验证循环 =====
        model.eval()
        val_metrics = defaultdict(float)
        val_preds = {t: [] for t in args.img_types}
        val_targets = {t: [] for t in args.img_types}
        
        with torch.no_grad(), tqdm(val_loader, desc=f"Rank {args.rank} Epoch {epoch+1}/{args.epochs} Val", 
                                  disable=not (args.rank == 0)) as pbar:
            for batch_idx, batch in enumerate(pbar):
                images = batch['image'].to(device, non_blocking=True)
                labels = {t: batch[t].to(device, non_blocking=True) for t in args.img_types}
                
                images = transforms.functional.normalize(
                    images, 
                    [0.485, 0.456, 0.406], 
                    [0.229, 0.224, 0.225]
                )
                
                # 验证阶段使用混合精度加速
                with torch.cuda.amp.autocast():
                    outputs, _ = model(images, task=None)
                    task_losses = []
                    for task in args.img_types:
                        task_loss = criterion[task](outputs[task], labels[task])
                        task_losses.append(task_loss)
                    
                    total_loss = awl(task_losses)
                
                for task in args.img_types:
                    probs = torch.softmax(outputs[task], dim=1)[:, 1].cpu().numpy()
                    val_preds[task].extend(probs)
                    val_targets[task].extend(labels[task].cpu().numpy())
                
                for t in args.img_types:
                    acc = (outputs[t].argmax(1) == labels[t]).float().mean().item()
                    val_metrics[f"{t}_acc"] += acc
                    val_metrics[f"{t}_loss"] += task_losses[args.img_types.index(t)].item()
                
                val_metrics["total_loss"] += total_loss.item()
                pbar.set_postfix({
                    "Loss": f"{val_metrics['total_loss'] / (batch_idx + 1):.4f}",
                    **{f"{t} Acc": f"{val_metrics[f'{t}_acc'] / (batch_idx + 1):.4f}"
                    for t in args.img_types}
                })

                # 专家内存整理
                if hasattr(model, 'module'):
                    model_blocks = model.module.blocks
                else:
                    model_blocks = model.blocks
                    
                for blk in model_blocks:
                    if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'step_counter'):
                        blk.mlp.step_counter += 1
                        if blk.mlp.step_counter % 25 == 0:
                            blk.mlp._defragment_memory()

                    if hasattr(blk, 'attn') and hasattr(blk.attn, 'step_counter'):
                        blk.attn.step_counter += 1
                        if blk.attn.step_counter % 25 == 0:
                            blk.attn._defragment_memory()

                # 显存清理
                if batch_idx % 5 == 0:
                    torch.cuda.empty_cache()
        
        val_auc = {task: calculate_auc(val_preds[task], val_targets[task], task)
                   for task in args.img_types}
        avg_val_auc = np.mean([val_auc[t] for t in args.img_types])
        
        # ===== 分布式聚合指标 =====
        if args.distributed:
            # 将所有进程的指标聚合到主进程
            avg_val_auc_tensor = torch.tensor(avg_val_auc).to(device)
            dist.reduce(avg_val_auc_tensor, dst=0, op=dist.ReduceOp.SUM)
            if args.rank == 0:
                avg_val_auc = avg_val_auc_tensor.item() / args.world_size
        
        # ===== 模型保存 =====
        if args.rank == 0:
            # 保存最佳模型（基于平均验证AUC）
            if avg_val_auc > best_val_auc:
                best_val_auc = avg_val_auc
                model_state = model.module.state_dict() if args.distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'awl_state_dict': awl.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_auc,
                }, os.path.join(save_dir, f'best_model.pth'))
                logger.info(f"保存最佳模型，平均验证AUC: {avg_val_auc:.4f}")
            
            # 定期保存检查点
            if (epoch + 1) % 10 == 0:
                model_state = model.module.state_dict() if args.distributed else model.state_dict()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'awl_state_dict': awl.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            
            # ===== 日志记录 =====
            # TensorBoard日志
            writer.add_scalar("Loss/train", train_metrics['total_loss']/len(train_loader), epoch)
            writer.add_scalar("Loss/val", val_metrics['total_loss']/len(val_loader), epoch)
            
            for t in args.img_types:
                writer.add_scalars(f"Accuracy/{t}", {
                    "train": train_metrics[f"{t}_acc"]/len(train_loader),
                    "val": val_metrics[f"{t}_acc"]/len(val_loader)
                }, epoch)
                
                writer.add_scalars(f"AUC/{t}", {
                    "train": train_auc[t],
                    "val": val_auc[t]
                }, epoch)
            
            # 文本日志
            log_msg = (f"Epoch {epoch+1}/{args.epochs} | "
                       f"Train Loss: {train_metrics['total_loss']/len(train_loader):.4f} | "
                       f"Val Loss: {val_metrics['total_loss']/len(val_loader):.4f} | "
                       f"Avg Val AUC: {avg_val_auc:.4f}\n")
            
            for t in args.img_types:
                log_msg += (f"{t}: "
                            f"Train Acc: {train_metrics[f'{t}_acc']/len(train_loader):.4f} "
                            f"Val Acc: {val_metrics[f'{t}_acc']/len(val_loader):.4f} | "
                            f"Train AUC: {train_auc[t]:.4f} Val AUC: {val_auc[t]:.4f}\n")
            
            logger.info(log_msg)
            print(log_msg)
    
    if args.rank == 0:
        writer.close()
        print(f"训练完成! 最佳验证AUC: {best_val_auc:.4f}")
    
    if args.distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Task ChestX-ray Classification")
    # 分布式参数
    parser.add_argument('--dist_url', default='env://', 
                        help='分布式训练初始化方法')
    parser.add_argument('--dist_backend', default='nccl', 
                        help='分布式后端')
   
    # 训练参数
    parser.add_argument("--dataset", type=str, default=r"datasets") ##########################################################数据集地址
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="mtvit_taskgate_att_mlp_base_MI_twice")
    parser.add_argument("--nb_classes", type=int, default=2)

    parser.add_argument("--img_types", nargs='+',
                        default=[
                            'class_Atelectasis', 'class_Cardiomegaly', 'class_Consolidation',
                            'class_Edema', 'class_Effusion', 'class_Emphysema',
                            'class_Fibrosis', 'class_Hernia', 'class_Infiltration',
                            'class_Mass', 'class_No_Finding', 'class_Nodule',
                            'class_Pleural_Thickening', 'class_Pneumonia', 'class_Pneumothorax'
                        ])

    parser.add_argument('--drop_path', type=float, default=0.1)
    parser.add_argument('--global_pool', action='store_true', default=True)
    parser.add_argument('--accum_iter', type=int, default=16)
    parser.add_argument('--cycle', action='store_true')
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--deepspeed', action='store_true',
                    help='Enable DeepSpeed ZeRO-3 training')
    parser.add_argument('--deepspeed_config', type=str, default=None,
                    help='Path to DeepSpeed JSON config')
    args = parser.parse_args()
    set_seed(args.seed)
    main(args)
