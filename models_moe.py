from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pos_embed import get_2d_sincos_pos_embed
from timm.layers import DropPath, to_2tuple
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from parallel_experts import MoE
from torch.amp import autocast
from torch import einsum
from einops import rearrange, repeat, pack, unpack
from parallel_experts import GSEMI2MoE, TaskMoE,Expert,GlobalSharedExperts
import math

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, head_dim=None, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(inner_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # For rare cases, the attention weights are inf due to the mix-precision training.
        # We clamp the tensor to the max values of the current data type
        # This is different from MAE training as we don't observe such cases on image-only MAE.
        if torch.isinf(attn).any():
            clamp_value = torch.finfo(attn.dtype).max-1000
            attn = torch.clamp(attn, min=-clamp_value, max=clamp_value)

        attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MoETaskAttention(nn.Module):
    def __init__(
        self,
        dim,
        noisy_gating=True,
        task_num=9,
        num_attn_experts=24,
        num_heads=8,
        head_dim=None,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        sample_topk=2,
        cvloss=0,
        switchloss=0.01 * 10,
        zloss=0.001 * 1,
        w_topk_loss=0.1,
        w_MI=0.,
        limit_k=0,
        moe_type='normal',
        shared_expert_ratio=0.25,
        private_dropout=0.2,
        contrast_weight=0.5,
        disentangle_weight=0.3
    ):
        super().__init__()
        self.task_num = task_num
        self.num_attn_experts = num_attn_experts
        self.sample_topk = sample_topk
        self.num_heads = num_heads
        if head_dim is None:
            head_dim = dim // num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.moe_type = moe_type

        self.private_dropout = nn.Dropout(private_dropout)
        self.contrast_weight = contrast_weight
        self.disentangle_weight = disentangle_weight

        # ---------- 共享专家 ----------
        attn_num_shared = max(1, int(num_attn_experts * shared_expert_ratio)) \
                          if shared_expert_ratio > 0 else 0
        attn_global_shared = nn.ModuleList([
            Expert(dim, head_dim, dim) for _ in range(attn_num_shared)
        ])
        # ---------- 调用 GSEMI2MoE ----------
        self.q_proj = GSEMI2MoE(
            input_size=dim,
            head_size=head_dim,
            num_experts=num_attn_experts,
            global_shared_experts=attn_global_shared,
            k=num_heads,
            num_shared_experts=attn_num_shared,
            noisy_gating=noisy_gating,
            w_MI=w_MI,
            acc_aux_loss=True,
            task_num=task_num,
            cvloss=cvloss,
            switchloss=switchloss,
            zloss=zloss,
            w_topk_loss=w_topk_loss,
            limit_k=limit_k,
            shared_expert_ratio=shared_expert_ratio,
            private_dropout=private_dropout,
            contrast_weight=contrast_weight
        )

        self.kv_proj = nn.Sequential(nn.Linear(dim, head_dim * 2))
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, task_bh, mask=None):
        B, N, C = x.shape
        q, aux_loss = self.q_proj.map(x, task_bh, sample_topk=self.sample_topk)
        k, v = self.kv_proj(x).chunk(2, dim=-1)
        q = q.reshape(B, N, self.num_heads, self.head_dim)
        k = k.reshape(B, N, self.head_dim)
        v = v.reshape(B, N, self.head_dim)

        attn = torch.einsum('bihd,bjd->bhij', q, k) * self.scale
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        if torch.isinf(attn).any():
            attn = torch.clamp(attn, min=-torch.finfo(attn.dtype).max+1000,
                                      max=torch.finfo(attn.dtype).max-1000)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn = torch.einsum('bhij,bjd->bihd', attn, v)
        x = self.q_proj.reduce(attn)
        x = self.proj_drop(x)
        return x, aux_loss

class MoEnhanceTaskBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        num_attn_experts=12,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        proj_drop=0.,
        num_ffd_experts=8,
        ffd_heads=2,
        ffd_noise=True,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        head_dim=None,
        init_values=None,
        z_weight=0.,
        post_layer_norm=False,
        task_num=9,
        noisy_gating=True,
        att_w_topk_loss=0.0,
        att_limit_k=0,
        cvloss=0,
        switchloss=0.01,
        zloss=0.001,
        w_topk_loss=0.0,
        limit_k=0,
        w_MI=0.,
        global_shared_experts=None,
        use_moe_mlp=True,
        use_moe_attn=True,
        shared_expert_ratio=0.1,
        num_shared_experts=0,
        sample_topk=0,
        private_dropout=0.2,
        contrast_weight=0.5,
        disentangle_weight=0.3,
        moe_type='normal'
    ):
        super().__init__()
        self.task_num = task_num
        self.norm1 = norm_layer(dim)
        self.use_moe_attn = use_moe_attn
        if use_moe_attn:
            # Attention 部分已在上方 fix
            self.attn = MoETaskAttention(
                dim=dim,
                task_num=task_num,
                num_attn_experts=num_attn_experts,
                num_heads=num_heads,
                head_dim=head_dim,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                sample_topk=sample_topk,
                cvloss=cvloss,
                switchloss=switchloss,
                zloss=zloss,
                w_topk_loss=att_w_topk_loss,
                w_MI=w_MI,
                limit_k=att_limit_k,
                moe_type=moe_type,
                shared_expert_ratio=shared_expert_ratio,
                private_dropout=private_dropout,
                contrast_weight=contrast_weight,
            )
        else:
            self.attn = Attention(dim, num_heads=num_heads, head_dim=head_dim,
                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                  attn_drop=attn_drop, proj_drop=proj_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.use_moe_mlp = use_moe_mlp
        if use_moe_mlp:
            # ---------- MLP 共享专家 ----------
            ffd_num_shared = max(1, int(num_ffd_experts * shared_expert_ratio)) \
                             if shared_expert_ratio > 0 else 0
            ffd_global_shared = nn.ModuleList([
                Expert(dim, mlp_hidden_dim // ffd_heads, dim)
                for _ in range(ffd_num_shared)
            ])
            self.mlp = GSEMI2MoE(
                input_size=dim,
                head_size=mlp_hidden_dim // ffd_heads,
                num_experts=num_ffd_experts,
                global_shared_experts=ffd_global_shared,
                k=ffd_heads,
                num_shared_experts=ffd_num_shared,
                bias=True,
                acc_aux_loss=True,
                cvloss=cvloss,
                switchloss=switchloss,
                zloss=zloss,
                w_topk_loss=w_topk_loss,
                w_MI=w_MI,
                limit_k=limit_k,
                task_num=task_num,
                activation=nn.GELU(),
                noisy_gating=ffd_noise,
                shared_expert_ratio=shared_expert_ratio,
                private_dropout=private_dropout,
                contrast_weight=contrast_weight
            )
        else:
            self.mlp = Mlp(dim, hidden_features=dim * 4, drop=drop)

    def forward(self, x, task_bh, mask=None):
        z_loss, aux_loss = 0., 0.
        if self.use_moe_attn:
            y, z_loss = self.attn(self.norm1(x), task_bh, mask=mask)
            x = x + self.drop_path(y)
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), mask=mask))

        if self.use_moe_mlp:
            y, aux_loss = self.mlp(self.norm2(x), task_bh)
            x = x + self.drop_path(y)
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, z_loss + aux_loss