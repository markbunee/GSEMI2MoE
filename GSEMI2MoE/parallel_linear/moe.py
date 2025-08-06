import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .parallel_experts import ParallelExperts

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor,
                  num_shared_experts: int):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)
    
    # 确保每个样本都被分配给k个专家
    batch_size = probs.size(0)
    expected_assignments = batch_size * k
    
    # 计算实际分配数
    actual_assignments = (gates > 0).sum().item()
    
    # 如果分配不足，添加缺失的分配
    if actual_assignments < expected_assignments:
        missing = expected_assignments - actual_assignments
        
        # 找到未被分配的样本
        unassigned_mask = (gates.sum(1) == 0)
        unassigned_indices = unassigned_mask.nonzero().squeeze(-1)
        
        # 为这些样本分配专家
        for i in range(unassigned_indices.size(0)):
            idx = unassigned_indices[i]
            # 首先尝试分配到共享专家
            shared_probs = probs[idx, :num_shared_experts]
            if shared_probs.sum() > 0:
                # 选择概率最高的共享专家
                expert_idx = shared_probs.argmax()
                min_gate = top_k_gates[top_k_gates > 0].min().item()
                gates[idx, expert_idx] = min_gate
            else:
                # 如果没有合适的共享专家，分配到私有专家
                private_probs = probs[idx, num_shared_experts:]
                if private_probs.sum() > 0:
                    # 选择概率最高的私有专家
                    expert_idx = private_probs.argmax() + num_shared_experts
                    min_gate = top_k_gates[top_k_gates > 0].min().item()
                    gates[idx, expert_idx] = min_gate
                else:
                    # 如果所有专家概率都为0，随机分配一个专家
                    # 使用张量索引而不是标量
                    expert_idx = torch.randint(0, probs.size(1), (1,))
                    gates[idx, expert_idx] = 1e-4
    
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    nonzeros = top_k_gates.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts[nonzeros]
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)
    index_sorted_experts = nonzeros[_index_sorted_experts]
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates[index_sorted_experts]
    
    return batch_gates, batch_index, expert_size, gates, index_sorted_experts



class MoE(nn.Module):
    def __init__(self, input_size, head_size, num_experts, k,
                 cvloss=0, switchloss=0, zloss=0,
                 bias=False, gating_activation=None,
                 activation=None, noisy_gating=True, usage_mem = 10000,
                 acc_aux_loss=False):
        super(MoE, self).__init__()

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.cvloss = cvloss
        self.switchloss = switchloss
        self.zloss = zloss
        self.activation = activation
        # self.usage = np.random.randint(num_experts, size=(usage_mem, k))
        # self.cur = 0


        self.acc_aux_loss = acc_aux_loss
        if self.acc_aux_loss:
            self.init_aux_statistics()

        if True:
            if gating_activation is None:
                gating_activation = nn.ReLU()
            self.f_gate = nn.Sequential(
                # nn.Linear(input_size, input_size),
                # gating_activation,
                nn.Linear(input_size,
                          2 * num_experts if noisy_gating else num_experts,
                          bias=False)
            )
            nn.init.zeros_(self.f_gate[-1].weight)
        else:
            self.f_gate = nn.Linear(input_size, num_experts, bias=False)
            nn.init.zeros_(self.f_gate.weight)


    def extra_repr(self):
        return 'k={}, cvloss={}, switchloss={}, zloss={}, noisy_gating={}'.format(
            self.k, self.cvloss, self.switchloss, self.zloss, self.noisy_gating)

    def cv_squared(self, x):

        eps = 1e-10
        # if only num_experts = 1

        if x.shape[0] == 1:
            return 0
        return x.float().var() / (x.float().mean()**2 + eps)

    def init_aux_statistics(self):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

    def update_aux_statistics(self, logits, probs, gates):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.000001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

    def get_aux_loss_and_clear(self):
        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        # cvloss = self.acc_gates.mean() / 10000.0
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)
        # loss = (self.cvloss * cvloss)
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss)
        self.init_aux_statistics()
        return loss
    def compute_cvloss(self, probs):
        return self.cv_squared(F.normalize(probs.sum(0), p=1, dim=0))

    def compute_switchloss(self, probs, freqs):
        loss = F.normalize(probs.sum(0), p=1, dim=0) * \
               F.normalize(freqs.float(), p=1, dim=0)
        return loss.sum() * self.num_experts

    def compute_zloss(self, logits):
        zloss = torch.mean(torch.log(torch.exp(logits).sum(dim=1)) ** 2)
        return zloss

    def top_k_gating(self, x, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):

        clean_logits = self.f_gate(x)
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1)
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        assert sample_topk == 0
        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]


        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices, 0)


        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            # if self.training:
            self.update_aux_statistics(logits, probs, gates)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate(x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y

    def map(self, x, skip_mask=None, sample_topk=0):

        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, skip_mask, sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y


class TaskMoE(MoE):
    def __init__(self,  input_size, head_size, num_experts, k, w_MI=0, limit_k=0, w_topk_loss=0.0, task_num=9, noisy_gating=True, gating_activation=None, **kwargs):
        self.task_num = task_num
        self.w_topk_loss = w_topk_loss
        self.w_MI = w_MI

        self.limit_k = max(k, limit_k)

        super(TaskMoE, self).__init__(input_size, head_size, num_experts, k, noisy_gating=noisy_gating, gating_activation=gating_activation, **kwargs)
        
        if gating_activation is None:
            gating_activation = nn.ReLU()

        self.f_gate = nn.ModuleList([nn.Sequential(
                                        # nn.Linear(input_size, input_size),
                                        # gating_activation,
                                        nn.Linear(input_size,
                                                  2 * num_experts if noisy_gating else num_experts,
                                                  bias=False)
                                    ) for i in range(task_num)])
        for i in range(task_num):
            nn.init.zeros_(self.f_gate[i][-1].weight)
    
    def init_aux_statistics(self, clear=True):
        self.acc_probs = 0.
        self.acc_gates = 0.
        self.acc_freq = 0.
        self.acc_lsesq = 0.
        self.acc_lsesq_count = 0.

        if clear:
            self.task_gate_freq = [0] * self.task_num
            self.topk_acc_probs = 0.

        self.MI_task_gate = torch.zeros(self.task_num, self.num_experts).cuda()

    def update_aux_statistics(self, logits, probs, gates, task_bh):
        lsesq = torch.log(torch.exp(logits).sum(dim=1) + 0.0001) ** 2
        self.acc_probs = self.acc_probs + probs.sum(0)
        self.acc_gates = self.acc_gates + gates.sum(0)
        self.acc_freq = self.acc_freq + (gates > 0).float().sum(0)
        self.acc_lsesq = self.acc_lsesq + lsesq.sum()
        self.acc_lsesq_count = self.acc_lsesq_count + lsesq.size(0)

        self.topk_acc_probs = self.topk_acc_probs + probs.mean(0)

        self.task_gate_freq[task_bh] = self.task_gate_freq[task_bh]*0.95 + ((gates > 0).float().sum(0)).detach()*0.05

        # self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + gates.sum(0)
        # self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh] + probs.sum(0)
        self.MI_task_gate[task_bh] = self.MI_task_gate[task_bh].cpu() + probs.sum(0).cpu()

    def get_topk_loss_and_clear(self):
        top_k_probs, top_k_indices = self.topk_acc_probs.topk(self.limit_k, dim=0)
        zeros = torch.zeros_like(self.topk_acc_probs)
        gates = zeros.scatter(0, top_k_indices, top_k_probs)
        topk_loss = ((self.topk_acc_probs - gates) * (self.topk_acc_probs - gates)).sum()

        self.topk_acc_probs = 0.
        return topk_loss * self.w_topk_loss # 0.004 * 12 * 2 = 0.09

    def get_aux_loss_and_clear(self):

        cvloss = self.cv_squared(F.normalize(self.acc_gates, p=1, dim=0))
        switchloss = (F.normalize(self.acc_probs, p=1, dim=0) *
                      F.normalize(self.acc_freq, p=1, dim=0)).sum() * self.num_experts
        zloss = self.acc_lsesq / (self.acc_lsesq_count)

        tot = self.acc_freq.sum() / self.k
        self.MI_task_gate = self.MI_task_gate / (tot+0.0001)
        P_TI = torch.sum(self.MI_task_gate, dim=1, keepdim=True) + 0.0001
        P_EI = torch.sum(self.MI_task_gate, dim=0, keepdim=True) + 0.0001

        MI_loss = -(self.MI_task_gate * torch.log(self.MI_task_gate / P_TI / P_EI + 0.0001)).sum()
        
        loss = (self.cvloss * cvloss +
                self.switchloss * switchloss +
                self.zloss * zloss +
                self.w_MI * MI_loss
                )

        self.init_aux_statistics(clear=False)
        return loss

    def top_k_gating(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):

        clean_logits = self.f_gate[task_bh](x)
        # if self.noisy_gating and self.training:
        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            noisy_logits = clean_logits + eps * noise_stddev
            logits = noisy_logits
        elif self.noisy_gating:
            logits, _ = clean_logits.chunk(2, dim=-1)
        else:
            logits = clean_logits

        probs = torch.softmax(logits, dim=1) + 1e-4
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        if self.training and (sample_topk > 0):
            # top_k_indices = torch.multinomial(probs + 1e-6, self.k)
            # top_k_gates = torch.gather(probs, 1, top_k_indices)
            assert sample_topk <= self.k

            _, top_km1_indices = probs.topk(self.k - sample_topk, dim=1)
            masked_probs = probs + 1e-6
            masked_probs[torch.arange(probs.size(0)).unsqueeze(
                1), top_km1_indices] = 0
            k_indices = torch.multinomial(masked_probs, sample_topk)
            top_k_indices = torch.cat([top_km1_indices, k_indices], dim=-1)
            top_k_gates = torch.gather(probs, 1, top_k_indices)
        else:
            top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

       # top_k_indecis: [batch, K]
       

        top_k_gates = top_k_gates

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices, self.num_shared_experts)
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, task_bh)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, self.expert_size)
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # y_ = self.forward_(x, skip_mask, sample_topk, multiply_by_gates)
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        # assert torch.allclose(y, y_)
        return y, loss

    def forward_(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # FOR DEBUGGING: naive forward
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        logits = self.f_gate[task_bh](x)
        probs = torch.softmax(logits, dim=1)

        loss = self.top_k_gating(x, skip_mask, task_bh, sample_topk=sample_topk)
        assert not self.bias
        hs = [torch.einsum('li,ij->lj', x, self.experts.w[i]) for i in range(self.num_experts)]
        hs = [self.activation(h) for h in hs]
        expert_outputs = [
            torch.einsum('li,ij->lj', hs[i], self.output_experts.w[i]) for i in range(self.num_experts)
        ]
        y = sum(probs[..., i][..., None] * expert_outputs[i] for i in range(self.num_experts))
        y = y.view(bsz, length, self.input_size)

        # if multiply_by_gates:
        #     expert_outputs = expert_outputs * self.batch_gates[:, None]
        return y


    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Args:
        x: tensor shape [batch_size, input_size]
        train: a boolean scalar.
        loss_coef: a scalar - multiplier on load-balancing losses
        Returns:
        y: a tensor with shape [batch_size, output_size].
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)
        loss = self.top_k_gating(x, task_bh, skip_mask,  sample_topk=sample_topk)

        expert_inputs = x[self.batch_index]
        # print('batch_index: ', batch_index)
        # print('expert_inputs: ', expert_inputs)
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)
        return y, loss

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y




class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.act = activation
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GlobalSharedExperts(nn.Module):
    """仅共享专家全局复用"""
    def __init__(self, input_size, hidden_size, num_shared):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_size, hidden_size, input_size) for _ in range(num_shared)
        ])
        self.num_shared = num_shared

class AdaptiveDisentangledGating(nn.Module):
    """
    低秩解耦门控：
    1) 先把 x 压缩到 rank 维
    2) 再用单一线性层输出 2 维权重 (shared, private)
    3) 保证 shared + private = 1
    """
    def __init__(self, input_size, rank=64):
        super().__init__()
        self.down = nn.Linear(input_size, rank, bias=False)   # 降维
        self.up   = nn.Linear(rank, 2, bias=False)            # 输出两维权重

    def forward(self, x):
        z = torch.relu(self.down(x))          # [B, rank]
        w = torch.sigmoid(self.up(z))         # [B, 2]  ∈ (0,1)
        s_weighted, p_weighted = w[:, 0:1], w[:, 1:2]

        # 归一化确保和为 1
        total = s_weighted + p_weighted + 1e-6
        s_weighted = s_weighted / total
        p_weighted = p_weighted / total
        return s_weighted, p_weighted

class ContrastiveLoss(nn.Module):
    """对比损失强化任务特异性"""
    def __init__(self, margin=1.0, temperature=0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        
    def forward(self, private_feat, task_ids):
        """
        Args:
            private_feat: 私有特征 [batch, features]
            task_ids: 任务ID [batch]
        """
        # 私有特征对比损失
        norm_private = F.normalize(private_feat, p=2, dim=1)
        sim_matrix = torch.mm(norm_private, norm_private.t()) / self.temperature
        
        # 创建任务标签矩阵
        task_mask = (task_ids.unsqueeze(0) == task_ids.unsqueeze(1)).float()
        
        # 正样本对（相同任务）
        positive_loss = -torch.log(torch.exp(sim_matrix) * task_mask + 1e-6).sum(1) / (task_mask.sum(1) + 1e-6)
        
        # 负样本对（不同任务）
        negative_loss = torch.log(torch.exp(sim_matrix) * (1 - task_mask) + 1e-6).sum(1) / ((1 - task_mask).sum(1) + 1e-6)
        
        contrastive_loss = (positive_loss + negative_loss).mean()
        
        return contrastive_loss


import gc
        
class GSEMI2MoE(TaskMoE):
    def __init__(
        self,
        input_size,
        head_size,
        num_experts,
        global_shared_experts: nn.ModuleList,
        k,
        num_shared_experts=0,
        shared_expert_ratio=0.0,
        shared_gate_noise=0.1,
        w_shared_loss=0.01,
        task_num=9,
        private_dropout=0.2,
        contrast_weight=0.5,
        **kwargs        
    ):

        if num_shared_experts > 0:
            self.num_shared_experts = min(num_shared_experts, num_experts)
        elif shared_expert_ratio > 0:
            self.num_shared_experts = max(1, int(num_experts * shared_expert_ratio))
        else:
            self.num_shared_experts = 0

        self.num_task_specific_experts = num_experts - self.num_shared_experts

        assert len(global_shared_experts) == self.num_shared_experts, \
            f"global_shared_experts 数量应为 {self.num_shared_experts}"

        super().__init__(
            input_size, head_size, num_experts, k,
            task_num=task_num, **kwargs
        )

                # 新增显存优化控制参数
        self.mem_optim_freq = 100  # 每100步执行一次显存优化
        self.step_counter = 0      # 训练步数计数器
        
        # 共享专家量化状态
        self.quantized = False

        self.shared_experts = nn.ParameterList([
            nn.Parameter(torch.cat([expert.fc1.weight for expert in global_shared_experts])) 
        ])  # 合并第一层权重 [2,4](@ref)

        self.private_experts = nn.ModuleList([
            Expert(input_size, head_size, input_size)
            for _ in range(self.num_task_specific_experts)
        ])
        self.shared_router = nn.Linear(input_size, self.num_shared_experts)
        nn.init.zeros_(self.shared_router.weight)

        self.shared_gate_noise = shared_gate_noise
        self.w_shared_loss = w_shared_loss
        self.contrast_weight = contrast_weight
        self.disentangle_gate = AdaptiveDisentangledGating(input_size)
        self.private_dropout = nn.Dropout(private_dropout)
        self.contrast_loss = ContrastiveLoss()
        if self.num_shared_experts > 0:
            self.shared_gate = nn.Linear(input_size, self.num_shared_experts, bias=False)
            nn.init.zeros_(self.shared_gate.weight)

    def top_k_gating_with_shared(self, x, task_bh, skip_mask=None, sample_topk=0, noise_epsilon=1e-2):
        clean_logits = self.f_gate[task_bh](x)

        if self.noisy_gating:
            clean_logits, raw_noise_stddev = clean_logits.chunk(2, dim=-1)
            noise_stddev = F.softplus(raw_noise_stddev) + noise_epsilon
            eps = torch.randn_like(clean_logits)
            logits = clean_logits + eps * noise_stddev
        else:
            logits = clean_logits

        if self.num_shared_experts > 0:
            shared_bias = self.shared_gate(x)
            if self.training:
                shared_bias = shared_bias + torch.randn_like(shared_bias) * self.shared_gate_noise
            logits[:, :self.num_shared_experts] += shared_bias

        probs = torch.softmax(logits, dim=1) + 1e-6
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)
            probs = probs / probs.sum(dim=1, keepdim=True)

        top_k_gates, top_k_indices = probs.topk(self.k, dim=1)

        gates = torch.zeros_like(probs)
        gates.scatter_(1, top_k_indices, top_k_gates)
        self.gates = gates

        loss = 0.
        if self.acc_aux_loss:
            self.update_aux_statistics(logits, probs, gates, task_bh)
        else:
            loss += self.cvloss * self.compute_cvloss(gates)
            loss += self.switchloss * \
                self.compute_switchloss(probs, (gates > 0).float().sum(0))
            loss += self.zloss * self.compute_zloss(logits)
        return loss

    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        # 在训练后期启用量化（不影响精度）
        if not self.training and not self.quantized and self.num_shared_experts > 0:
            self._quantize_shared_experts()
            
        with torch.cuda.amp.autocast():
            bsz, length, emb_size = x.size()
            x_flat = x.view(-1, emb_size)
            
            # 门控计算
            loss = self.top_k_gating_with_shared(x_flat, task_bh, skip_mask, sample_topk)
            gates = self.gates
            
            # 使用稀疏张量优化显存
            sparse_gates = gates.to_sparse()
            token_idx, expert_idx = sparse_gates.indices()
            gate_values = sparse_gates.values()
            
            # 动态构建专家桶
            expert_buckets = {}
            for idx, eid in enumerate(expert_idx.tolist()):
                expert_buckets.setdefault(eid, []).append(idx)

            # 特征解耦
            s_w, p_w = self.disentangle_gate(x_flat)
            y_flat = torch.zeros_like(x_flat, dtype=torch.float16)

            # 专家计算优化
            for eid, token_indices in expert_buckets.items():
                # 使用原生索引避免创建新张量
                token_indices = torch.tensor(token_indices, device=x_flat.device)
                token_subset = token_idx[token_indices]
                
                if eid < self.num_shared_experts:
                    # 动态计算输入特征（避免全量复制）
                    input_feat = s_w[token_subset] * x_flat[token_subset]
                    
                    # 共享专家参数动态加载
                    expert_weight = self.shared_experts[0][eid*self.head_size:(eid+1)*self.head_size]
                    out_subset = F.linear(input_feat, expert_weight)
                else:
                    # 私有专家计算
                    input_feat = p_w[token_subset] * x_flat[token_subset]
                    input_feat = self.private_dropout(input_feat)
                    
                    if self.training:
                        out_subset = torch.utils.checkpoint.checkpoint(
                            self.private_experts[eid - self.num_shared_experts], 
                            input_feat,
                            use_reentrant=False  # 减少显存保留
                        )
                    else:
                        out_subset = self.private_experts[eid - self.num_shared_experts](input_feat)
                
                # 原位操作减少中间变量
                weighted_out = gate_values[token_indices].unsqueeze(1) * out_subset
                y_flat.index_add_(0, token_subset, weighted_out.to(y_flat.dtype))
            
            y = y_flat.view(bsz, length, emb_size).to(x.dtype)

            # 对比损失计算优化
            if self.training:
                # 复用已计算的p_w避免重复计算
                private_x = p_w * x_flat
                contrast_loss = self.contrast_loss(
                    private_x,
                    torch.full((private_x.size(0),), task_bh, device=private_x.device)
                )
                loss += self.contrast_weight * contrast_loss

        # 精准显存清理策略
        self._memory_cleanup(
            gates, x_flat, y_flat, expert_buckets, 
            token_idx, expert_idx, gate_values,
            s_w, p_w, private_x if self.training else None
        )
        
        # 定期执行显存碎片整理
        self.step_counter += 1
        if self.step_counter % self.mem_optim_freq == 0:
            self._defragment_memory()
        
        return y, loss
    
    def _quantize_shared_experts(self):
        """推理时动态量化共享专家参数"""
        if self.quantized or self.num_shared_experts == 0:
            return
                
        with torch.no_grad():
            # 动态量化（减少75%显存）
            weight = self.shared_experts[0]
            scale = torch.max(torch.abs(weight)) / 127
            quantized_weight = torch.clamp(torch.round(weight / scale), -128, 127).to(torch.int8)
            # 转换为浮点类型
            quantized_weight = quantized_weight.float() * scale
            self.shared_experts[0] = nn.Parameter(quantized_weight)
            self.scale = scale  # 存储反量化比例
            self.quantized = True
    
    def _memory_cleanup(self, *args):
        """精准清理中间变量"""
        for tensor in args:
            if tensor is not None:
                del tensor
                
        # 清除PyTorch的缓存分配器保留的显存
        torch.cuda.empty_cache()
    
    def _defragment_memory(self):
        """显存碎片整理策略"""
        # 1. 清空CUDA缓存
        torch.cuda.empty_cache()
        
        # 2. 执行垃圾回收
        gc.collect()
        
        # 3. 限制进程显存使用比例（减少碎片）
        torch.cuda.set_per_process_memory_fraction(0.85)
        
        # 4. 记录显存状态（调试用）
        if self.step_counter % 500 == 0:
            print(f"[MemOptim] Step {self.step_counter} - "
                  f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB, "
                  f"Cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")