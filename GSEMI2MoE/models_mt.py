from functools import partial
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import os
import numpy as np
from models_moe import PatchEmbed,MoEnhanceTaskBlock
from parallel_experts import GSEMI2MoE, TaskMoE,Expert,GlobalSharedExperts
import timm.models.vision_transformer
class MTVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_types, embed_dim=768, global_pool=True, **kwargs):
        super(MTVisionTransformer, self).__init__(embed_dim=embed_dim, **kwargs)
        self.taskGating = False
        self.ismoe = False
      
        self.moe_type = 'normal'

        self.img_types = [type_ for type_ in img_types if type_ != 'rgb']
        assert global_pool == True
        del self.head
        norm_layer = kwargs['norm_layer']
        self.fc_norm = norm_layer(embed_dim)

        # create task head
        self.task_heads = []
        type_to_channel = {'depth_euclidean':1, 'depth_zbuffer':1, 'edge_occlusion':1, 'edge_texture':1, 'keypoints2d':1, 'keypoints3d':1, 'normal':3, 'principal_curvature':2,  'reshading':3, 'rgb':3, 'segment_semantic':18, 'segment_unsup2d':1, 'segment_unsup25d':1}
        image_height, image_width = self.patch_embed.img_size
        patch_height, patch_width = self.patch_embed.patch_size
        assert image_height == 224 and image_width == 224
        for t in range(len(self.img_types)): ###
            img_type = self.img_types[t]
            if 'class' in img_type:
                # class_num = 1000 if img_type == 'class_object' else 365
                # class_num = 2
                self.task_heads.append(
                        # Use the cls token
                        nn.Sequential(
                            nn.LayerNorm(embed_dim),
                            nn.Linear(embed_dim,2)
                        )
                    )
            else:
                channel = type_to_channel[img_type]
                self.task_heads.append(
                        # Use the other token
                        nn.Sequential(
                            Rearrange('b (h w) d -> (b h w) d', h = image_height//patch_height, w= image_width//patch_width),
                            nn.Linear(embed_dim, patch_height * patch_width * channel),
                            Rearrange('(b h w) (j k c) -> b (h j) (w k) c', h = image_height//patch_height, w = image_width//patch_width, j=patch_height, k=patch_width, c=channel),
                        )
                    )
        self.task_heads = nn.ModuleList(self.task_heads)

        self.task_embedding = nn.Parameter(torch.randn(1, len(self.img_types), embed_dim))

        self.apply(self._init_weights)

    def forward_features(self, x, task_rank, task):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.pos_drop(x)

        # apply Transformer blocks

        for blk in self.blocks:
            x = x + self.task_embedding[:,task_rank:task_rank+1, :]
            x = blk(x)

        if 'class' in task:
            x = x[:, 1:, :].mean(dim=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 1:, :]

        return x, 0

    def forward(self, x, task, get_flop=False):
        task_rank = -1
        for t, the_type in enumerate(self.img_types):
            if the_type == task:
                task_rank = t
                break
        assert task_rank > -1

        x, z_loss = self.forward_features(x, task_rank, task)
        x = self.task_heads[task_rank](x)

        if get_flop:
            return x
        return x, z_loss


# from models_vit import VisionTransformer

def move_dict(ckpt, src, tgt):
    if src in ckpt and (src!=tgt):
        ckpt[tgt] = ckpt[src]
        del ckpt[src]

# A gating for a task
class MTVisionTransformerMoETaskGating(MTVisionTransformer):
    def __init__(self, img_types, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None,
                 num_attn_experts=48, head_dim=None, att_w_topk_loss=0.0, att_limit_k=0,
                 num_ffd_experts=16, ffd_heads=2, ffd_noise=True,
                 moe_type='normal',
                 switchloss=0.01 * 1, zloss=0.001 * 1, w_topk_loss= 0.0, limit_k=0,
                 w_MI = 0.,
                 noisy_gating=True,
                 post_layer_norm=False,
                 twice_mlp=False,
                 twice_attn=False,shared_gate_noise=0.1,shared_expert_ratio=0.1, 
                 **kwargs):
        super(MTVisionTransformerMoETaskGating, self).__init__(img_types,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            **kwargs)

        self.moe_type = moe_type
        self.depth = depth

        self.w_topk_loss = w_topk_loss
        self.taskGating = True
        self.ismoe = True
        self.task_num = len(self.img_types)
        self.R = {
                'depth': depth,
                'task_num': self.task_num,
                'head_dim': head_dim,
                'noisy_gating': noisy_gating,
                'ffd_heads': ffd_heads, 'ffd_noise': ffd_noise,
                'dim': embed_dim, 'num_heads': num_heads, 'mlp_ratio': mlp_ratio, 'qkv_bias': qkv_bias,
                'drop': drop_rate, 'attn_drop': attn_drop_rate, 'drop_path_rate': drop_path_rate, 'norm_layer': norm_layer,
                'moe_type': moe_type, 'switchloss': switchloss, 'zloss': zloss, 'w_topk_loss': w_topk_loss, 'limit_k': limit_k,
                'post_layer_norm': post_layer_norm,
                'twice_mlp': twice_mlp,
                'twice_attn': twice_attn,
                }

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.twice_mlp = twice_mlp

        # 计算共享专家数量
        num_shared = max(1, int(num_ffd_experts * shared_expert_ratio)) \
                    if shared_expert_ratio > 0 else 0

        # 创建共享专家池（容器）
        global_shared = nn.ModuleList([
            Expert(embed_dim, embed_dim // ffd_heads, embed_dim)
            for _ in range(num_shared)
        ])
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]


        self.blocks = nn.Sequential(*[
            MoEnhanceTaskBlock(
                task_num=self.task_num,
                num_attn_experts=num_attn_experts, head_dim=head_dim,
                num_ffd_experts=num_ffd_experts, ffd_heads=ffd_heads, ffd_noise=ffd_noise,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                moe_type=moe_type,switchloss=switchloss, zloss=zloss, w_topk_loss=w_topk_loss, limit_k=limit_k,
                w_MI = w_MI,num_shared_experts=num_shared, global_shared_experts=global_shared,
                noisy_gating=noisy_gating,
                att_w_topk_loss=att_w_topk_loss, att_limit_k=att_limit_k,
                post_layer_norm=post_layer_norm,
                use_moe_mlp=(twice_mlp==False or (i%2)==1),
                use_moe_attn=(twice_attn==False or (i%2)==0),
                )
            for i in range(depth)])

        self.apply(self._init_weights)

    # reload
    def pruning(self, args):
        if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
            vis_file = '/gpfs/u/home/AICD/AICDzich/scratch/' + str(args.copy) + '_vis.t7'
            load_file = '/gpfs/u/home/AICD/AICDzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'
        else:
            vis_file = '/gpfs/u/home/LMCG/LMCGzich/scratch/' + str(args.copy) + '_vis.t7'
            load_file = '/gpfs/u/home/LMCG/LMCGzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'

        the_list = torch.load(vis_file)
        # print(the_list)
        all_experts = []

        dpr = [x.item() for x in torch.linspace(0, self.R['drop_path_rate'], self.R['depth'])]

        the_blocks = []
        # pruning_attn, pruning_mlp = False, False
        pruning_attn = [False] * self.depth
        pruning_mlp = [False] * self.depth
        for depth, blk in enumerate(self.blocks):
            expert_usage = the_list[depth][args.the_task] # a list of int for experts
            # mlp_bh = 1 if blk.attn.num_experts > blk.attn.num_heads else 0
            mlp_bh = 0

            num_attn_experts = blk.attn.num_heads
            if hasattr(blk.attn, 'num_experts'):
                if blk.attn.num_experts > blk.attn.num_heads:
                    mlp_bh = 1
                    choose = (np.array(expert_usage[0]) > args.thresh / blk.attn.num_heads)
                    num_attn_experts = int(choose.sum())

                    if num_attn_experts < blk.attn.num_heads: # threshold too large
                        ind = np.argpartition(np.array(expert_usage[0]), -blk.attn.num_heads)[-blk.attn.num_heads:]
                        choose[ind] = True
                        num_attn_experts = blk.attn.num_heads
                    pruning_attn[depth] = True
                else:
                    num_attn_experts = blk.attn.num_experts

            num_ffd_experts = 1
            if hasattr(blk.mlp, 'num_experts'):
                if blk.mlp.num_experts > blk.mlp.k:
                    choose = (np.array(expert_usage[mlp_bh]) > args.thresh / blk.mlp.k)
                    num_ffd_experts = int(choose.sum())

                    if num_ffd_experts < blk.mlp.k: # threshold too large
                        ind = np.argpartition(np.array(expert_usage[mlp_bh]), -blk.mlp.k)[-blk.mlp.k:]
                        choose[ind] = True
                        num_ffd_experts = blk.mlp.k
                    pruning_mlp[depth] = True
                else:
                    num_ffd_experts = blk.mlp.num_experts

            # print(args.the_task, depth, num_attn_experts, num_ffd_experts)
            # miss att_w_topk_loss
            the_blocks.append(
                MoEnhanceTaskBlock(
                        num_attn_experts=num_attn_experts, num_ffd_experts=num_ffd_experts,
                        drop_path=dpr[depth],
                        ffd_noise=self.R['ffd_noise'],
                        task_num=self.R['task_num'],
                        head_dim=self.R['head_dim'],
                        ffd_heads=self.R['ffd_heads'],
                        noisy_gating=self.R['noisy_gating'],
                        dim=self.R['dim'], num_heads=self.R['num_heads'], mlp_ratio=self.R['mlp_ratio'], qkv_bias=self.R['qkv_bias'],
                        drop=dpr[depth], attn_drop=self.R['attn_drop'], norm_layer=self.R['norm_layer'],
                        moe_type=self.R['moe_type'],switchloss=self.R['switchloss'], zloss=self.R['zloss'],
                        w_topk_loss=self.R['w_topk_loss'], limit_k=self.R['limit_k'],
                        post_layer_norm=self.R['post_layer_norm'],
                        use_moe_mlp=(self.R['twice_mlp']==False or (depth%2)==1),
                        use_moe_attn=(self.R['twice_attn']==False or (depth%2)==0),
                    )
                )

        del self.blocks
        self.blocks = nn.Sequential(*the_blocks)

        # Careful Here!!!
        # origin_task = ['class_object', 'class_scene', 'depth_euclidean', 'depth_zbuffer', 'normal', 'principal_curvature', 'reshading', 'segment_unsup2d', 'segment_unsup25d']
        # origin_task = args.ori_img_types
        origin_task = [type_ for type_ in args.ori_img_types if type_ != 'rgb']

        task_bh = -1
        for i, the_task in enumerate(origin_task):
            if the_task == args.the_task:
                task_bh = i
                break
        assert task_bh >= 0

        checkpoint_all = torch.load(load_file, map_location='cpu')
        checkpoint = checkpoint_all['model']

        delete_key = []
        for c_key in checkpoint.keys():
            the_key = 'f_gate.' + str(task_bh) + '.'
            if ('f_gate.' in c_key) and (the_key not in c_key):
                # print('delete ', c_key)
                delete_key.append(c_key)
        for c_key in delete_key:
            del checkpoint[c_key]

        # print(checkpoint.keys())
        for depth, blk in enumerate(self.blocks):
            expert_usage = the_list[depth][args.the_task]

            prefix = 'blocks.' + str(depth) + '.attn.q_proj.'

            if task_bh != -1:
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.weight', prefix+'f_gate.'+'0'+'.0.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.bias', prefix+'f_gate.'+'0'+'.0.bias')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.2.weight', prefix+'f_gate.'+'0'+'.2.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.2.bias', prefix+'f_gate.'+'0'+'.2.bias')

            # TaskMoe experts.w experts.b output_experts.w output_experts.b f_gate.task_bh.0

            # if hasattr(blk.attn, 'num_experts'):
            #     if blk.attn.num_experts > blk.attn.num_heads:
            # if pruning_attn:
            if pruning_attn[depth]:
                # select_id = (torch.from_numpy(np.array(expert_usage[0])) > args.thresh).nonzero().view(-1)

                select_id = (np.array(expert_usage[0]) > args.thresh / blk.attn.num_heads)
                num_attn_experts = int(select_id.sum())

                if num_attn_experts < blk.attn.num_heads: # threshold too large
                    ind = np.argpartition(np.array(expert_usage[0]), -blk.attn.num_heads)[-blk.attn.num_heads:]
                    select_id[ind] = True
                    num_attn_experts = blk.attn.num_heads
                select_id = torch.from_numpy(select_id).nonzero().view(-1)
                print('select_id: ', select_id)

                for words in ['experts.w', 'experts.b', 'output_experts.w', 'output_experts.b', 'f_gate.'+'0'+'.0.weight', 'f_gate.'+'0'+'.0.bias']:
                    the_key = prefix+words
                    if the_key in checkpoint:
                        # print(words, ' : ', checkpoint[the_key].shape)

                        if 'f_gate' not in the_key:
                            tgt_key = the_key
                        elif '.weight' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.weight'
                        elif '.bias' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.bias'

                        if blk.attn.q_proj.noisy_gating and 'f_gate' in words:
                            the_id = select_id + checkpoint[the_key].shape[0] // 2
                            the_id = torch.cat((select_id, the_id), 0)
                            # print('the_id: ', the_id, tgt_key)
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, the_id)
                        else:
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, select_id)
                            # print(tgt_key, select_id, checkpoint[tgt_key].shape)

            prefix = 'blocks.' + str(depth) + '.mlp.'
            if task_bh != -1:
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.weight', prefix+'f_gate.'+'0'+'.0.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.bias', prefix+'f_gate.'+'0'+'.0.bias')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.weight', prefix+'f_gate.'+'0'+'.0.weight')
                move_dict(checkpoint, prefix+'f_gate.'+str(task_bh)+'.0.bias', prefix+'f_gate.'+'0'+'.0.bias')


            # if pruning_mlp:
            # if hasattr(blk.mlp, 'num_experts'):
            #     if blk.mlp.num_experts > blk.mlp.k:
            if pruning_mlp[depth]:
                # select_id = (torch.from_numpy(np.array(expert_usage[mlp_bh])) > args.thresh).nonzero().view(-1)
                select_id = (np.array(expert_usage[mlp_bh]) > args.thresh / blk.mlp.k)
                num_ffd_experts = int(select_id.sum())

                if num_ffd_experts < blk.mlp.k: # threshold too large
                    ind = np.argpartition(np.array(expert_usage[mlp_bh]), -blk.mlp.k)[-blk.mlp.k:]
                    select_id[ind] = True
                    num_ffd_experts = blk.mlp.k
                select_id = torch.from_numpy(select_id).nonzero().view(-1)

                for words in ['experts.w', 'experts.b', 'output_experts.w', 'output_experts.b', 'f_gate.'+'0'+'.0.weight', 'f_gate.'+'0'+'.0.bias']:
                    the_key = prefix+words
                    if the_key in checkpoint:
                        # print(words, ' : ', checkpoint[the_key].shape)
                        # print('depth: ', depth, the_key, select_id)

                        if 'f_gate' not in the_key:
                            tgt_key = the_key
                        elif '.weight' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.weight'

                            # checkpoint[prefix+'f_gate.'+'0'+'.0.weight'] = checkpoint[prefix+'f_gate.'+str(task_bh)+'.0.weight']
                        elif '.bias' in the_key:
                            tgt_key = prefix+'f_gate.'+'0'+'.0.bias'

                            # checkpoint[prefix+'f_gate.'+'0'+'.0.bias'] = checkpoint[prefix+'f_gate.'+str(task_bh)+'.0.bias']

                        if blk.mlp.noisy_gating and 'f_gate' in words:
                            the_id = select_id + checkpoint[the_key].shape[0] // 2
                            the_id = torch.cat((select_id, the_id), 0)
                            # print('the_id: ', the_id, tgt_key)
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, the_id)
                            # print(checkpoint[tgt_key].shape)
                        else:
                            checkpoint[tgt_key] = torch.index_select(checkpoint[the_key], 0, select_id)

        src_key = 'task_heads.' + str(task_bh) + '.'
        tgt_key = 'task_heads.0.'
        new_dict = {}
        delete_key = []
        for c_key in checkpoint.keys():
            if src_key in c_key:
                new_dict[tgt_key + c_key[len(src_key):]] = checkpoint[c_key]
                # print(c_key, tgt_key + c_key[len(src_key):])
            if 'task_heads' in c_key:
                delete_key.append(c_key)

        for c_key in delete_key:
            del checkpoint[c_key]
        checkpoint.update(new_dict)

        if task_bh != -1:
            checkpoint['task_embedding'] = checkpoint['task_embedding'][:,task_bh:task_bh+1]
        else:
            del checkpoint['task_embedding']

        return checkpoint

    def delete_ckpt(self, args): # the_task is not in origin task
        if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
            load_file = '/gpfs/u/home/AICD/AICDzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'
        else:
            load_file = '/gpfs/u/home/LMCG/LMCGzich/scratch/work_dirs/MTMoe/' + str(args.copy) + '/use.pth'

        checkpoint_all = torch.load(load_file, map_location='cpu')
        checkpoint = checkpoint_all['model']

        delete_key = []
        for c_key in checkpoint.keys():
            if ('f_gate.' in c_key) or ('task_heads' in c_key):
                # print('delete ', c_key)
                delete_key.append(c_key)
        for c_key in delete_key:
            del checkpoint[c_key]

        del checkpoint['task_embedding']
        return checkpoint

    def frozen(self):
        self.patch_embed.requires_grad = False
        self.pos_embed.requires_grad = False
        self.cls_token.requires_grad = False
        for blk in self.blocks:
            blk.attn.kv_proj.requires_grad = False
            blk.attn.q_proj.experts.requires_grad = False
            blk.attn.q_proj.output_experts.requires_grad = False

            blk.mlp.experts.requires_grad = False
            blk.mlp.output_experts.requires_grad = False

    def moa_init_weight(self, module):
        if isinstance(module, (nn.Linear)):
            module.weight.data.fill_(0.00)

    def get_zloss(self):
        z_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_aux_loss_and_clear()
                z_loss = z_loss + aux_loss

            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_aux_loss_and_clear()
                z_loss = z_loss + aux_loss
        return z_loss

    def get_topkloss(self):
        z_loss = 0
        for blk in self.blocks:
            if hasattr(blk.attn, 'num_experts'):
                aux_loss = blk.attn.q_proj.get_topk_loss_and_clear()
                z_loss = z_loss + aux_loss

            # break
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.get_topk_loss_and_clear()
                z_loss = z_loss + aux_loss
        return z_loss

    def all_clear(self):
        for blk in self.blocks:
            aux_loss = blk.attn.q_proj.init_aux_statistics()
            if hasattr(blk.mlp, 'num_experts'):
                aux_loss = blk.mlp.init_aux_statistics()

    def visualize(self, vis_head=False, vis_mlp=False, model_name=''):
        all_list = []
        torch.set_printoptions(precision=2, sci_mode=False)

        for depth, blk in enumerate(self.blocks):
            layer_list = {}
            for i, the_type in enumerate(self.img_types):

                layer_list[the_type] = []
                if hasattr(blk.attn, 'num_experts'):
                    if blk.attn.num_experts > blk.attn.num_heads:
                        _sum = blk.attn.q_proj.task_gate_freq[i].sum()
                        layer_list[the_type].append((blk.attn.q_proj.task_gate_freq[i] / _sum * 100).tolist())
                        # print('L', depth, ' attn: ', blk.attn.q_proj.task_gate_freq[i] / _sum * 100)

                if hasattr(blk.mlp, 'num_experts'):
                    if blk.mlp.num_experts > blk.mlp.k:
                        _sum = blk.mlp.task_gate_freq[i].sum()
                        layer_list[the_type].append((blk.mlp.task_gate_freq[i] / _sum * 100).tolist())
                        # print('L', depth, ' mlp: ', blk.mlp.task_gate_freq[i] / _sum * 100)
            all_list.append(layer_list)
        print(all_list)

        if os.getcwd()[:26] == '/gpfs/u/barn/AICD/AICDzich' or os.getcwd()[:26] == '/gpfs/u/home/AICD/AICDzich':
            torch.save(all_list, '/gpfs/u/home/AICD/AICDzich/scratch/' + str(model_name) + '_vis.t7')
        else:
            torch.save(all_list, '/gpfs/u/home/LMCG/LMCGzich/scratch/' + str(model_name) + '_vis.t7')

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        # x = x + self.task_embedding[:,task_rank:task_rank+1, :]
        x_before = self.pos_drop(x)

        # apply Transformer blocks

        output = {}
        z_loss = 0
        for t, the_type in enumerate(self.img_types):
            x = x_before
      
            for blk in self.blocks:
                x = x + self.task_embedding[:, t:t+1, :]
                x, _ = blk(x, t)

            if 'class' in the_type:
                x = x[:, 1:, :].mean(dim=1)
                x = self.fc_norm(x)
            else:
                x = self.norm(x)
                x = x[:, 1:, :]
            output[the_type] = x
            if self.w_topk_loss > 0.0:
                z_loss = z_loss + self.get_topkloss()

        return output, z_loss

    def forward(self, x, task, get_flop=False, get_z_loss=False):

        output, z_loss = self.forward_features(x)
        for t, the_type in enumerate(self.img_types):
            output[the_type] = self.task_heads[t](output[the_type])

        if get_flop:
            return output['class_object']

        # self.all_clear()
        return output, z_loss + self.get_zloss()

def mtvit_taskgate_att_mlp_base_MI_twice(img_types, **kwargs): # number of params (M): 195.80
    model = MTVisionTransformerMoETaskGating(img_types,
        patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
        num_attn_experts=24, head_dim=768//12 * 2,
        num_ffd_experts=16, ffd_heads=4, ffd_noise=False, mlp_ratio=4,
        w_MI=0.005, switchloss=0.0, zloss=0.0,
        noisy_gating=False,
        twice_mlp=True,
        twice_attn=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

