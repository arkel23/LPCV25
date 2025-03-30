"""PyTorch LRResNet

This started as a copy of https://github.com/pytorch/vision 'resnet.py' (BSD-3-Clause) with
additional dropout and dynamic global avg/max pool.

ResNeXt, SE-ResNeXt, SENet, and MXNet Gluon stem/downsample variants, tiered stems added by Ross Wightman

Copyright 2019, Ross Wightman

Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""
import math
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import Final
from einops import repeat, rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import DropBlock2d, DropPath, AvgPool2dSame, BlurPool2d, GroupNorm, LayerType, create_attn, \
    get_attn, get_act_layer, get_norm_layer, create_classifier, Mlp, trunc_normal_, use_fused_attn, RmsNorm
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations


__all__ = ['LRResNet', 'BasicBlock', 'Bottleneck']  # model_registry will add each entrypoint fn to this


def get_padding(kernel_size: int, stride: int, dilation: int = 1) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def create_aa(aa_layer: Type[nn.Module], channels: int, stride: int = 2, enable: bool = True) -> nn.Module:
    if not aa_layer or not enable:
        return nn.Identity()
    if issubclass(aa_layer, nn.AvgPool2d):
        return aa_layer(stride)
    else:
        return aa_layer(channels=channels, stride=stride)


class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class TransformerBlock(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 0.5,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.norm1(self.attn(x)))
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        return x


class TransformerParallelScalingBlock(nn.Module):
    """ Parallel ViT block (MLP & Attention in parallel)
    Based on:
      'Scaling Vision Transformers to 22 Billion Parameters` - https://arxiv.org/abs/2302.05442
    """
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 0.25,
            qkv_bias: bool = False,
            qk_norm: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = RmsNorm,
            mlp_layer: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        mlp_hidden_dim = int(mlp_ratio * dim)
        in_proj_out_dim = mlp_hidden_dim + 3 * dim

        self.in_norm = norm_layer(dim)
        self.in_proj = nn.Linear(dim, in_proj_out_dim, bias=qkv_bias)
        self.in_split = [mlp_hidden_dim] + [dim] * 3
        if qkv_bias:
            self.register_buffer('qkv_bias', None)
            self.register_parameter('mlp_bias', None)
        else:
            self.register_buffer('qkv_bias', torch.zeros(3 * dim), persistent=False)
            self.mlp_bias = nn.Parameter(torch.zeros(mlp_hidden_dim))

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.attn_out_proj = nn.Linear(dim, dim)

        self.mlp_drop = nn.Dropout(proj_drop)
        self.mlp_act = act_layer()
        self.mlp_out_proj = nn.Linear(mlp_hidden_dim, dim)

        self.ls = LayerScale(dim, init_values=init_values) if init_values is not None else nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape

        # Combined MLP fc1 & qkv projections
        y = self.in_norm(x)
        if self.mlp_bias is not None:
            # Concat constant zero-bias for qkv w/ trainable mlp_bias.
            # Appears faster than adding to x_mlp separately
            y = F.linear(y, self.in_proj.weight, torch.cat((self.qkv_bias, self.mlp_bias)))
        else:
            y = self.in_proj(y)
        x_mlp, q, k, v = torch.split(y, self.in_split, dim=-1)

        # Dot product attention w/ qk norm
        q = self.q_norm(q.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(k.view(B, N, self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.fused_attn:
            x_attn = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_attn = attn @ v
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.attn_out_proj(x_attn)

        # MLP activation, dropout, fc2
        x_mlp = self.mlp_act(x_mlp)
        x_mlp = self.mlp_drop(x_mlp)
        x_mlp = self.mlp_out_proj(x_mlp)

        # Add residual w/ drop path & layer scale applied
        y = self.drop_path(self.ls(x_attn + x_mlp))
        x = x + y
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(BasicBlock, self).__init__()

        assert cardinality == 1, 'BasicBlock only supports cardinality of 1'
        assert base_width == 64, 'BasicBlock does not support changing base width'
        first_planes = planes // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(
            inplanes, first_planes, kernel_size=3, stride=1 if use_aa else stride, padding=first_dilation,
            dilation=first_dilation, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act1 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=first_planes, stride=stride, enable=use_aa)

        self.conv2 = nn.Conv2d(
            first_planes, outplanes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act2 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn2, 'weight', None) is not None:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.drop_block(x)
        x = self.act1(x)
        x = self.aa(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act2(x)

        return x


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            cardinality: int = 1,
            base_width: int = 64,
            reduce_first: int = 1,
            dilation: int = 1,
            first_dilation: Optional[int] = None,
            act_layer: Type[nn.Module] = nn.ReLU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            attn_layer: Optional[Type[nn.Module]] = None,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_block: Optional[Type[nn.Module]] = None,
            drop_path: Optional[nn.Module] = None,
    ):
        """
        Args:
            inplanes: Input channel dimensionality.
            planes: Used to determine output channel dimensionalities.
            stride: Stride used in convolution layers.
            downsample: Optional downsample layer for residual path.
            cardinality: Number of convolution groups.
            base_width: Base width used to determine output channel dimensionality.
            reduce_first: Reduction factor for first convolution output width of residual blocks.
            dilation: Dilation rate for convolution layers.
            first_dilation: Dilation rate for first convolution layer.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            attn_layer: Attention layer.
            aa_layer: Anti-aliasing layer.
            drop_block: Class for DropBlock layer.
            drop_path: Optional DropPath layer.
        """
        super(Bottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        first_planes = width // reduce_first
        outplanes = planes * self.expansion
        first_dilation = first_dilation or dilation
        use_aa = aa_layer is not None and (stride == 2 or first_dilation != dilation)

        self.conv1 = nn.Conv2d(inplanes, first_planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(first_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(
            first_planes, width, kernel_size=3, stride=1 if use_aa else stride,
            padding=first_dilation, dilation=first_dilation, groups=cardinality, bias=False)
        self.bn2 = norm_layer(width)
        self.drop_block = drop_block() if drop_block is not None else nn.Identity()
        self.act2 = act_layer(inplace=True)
        self.aa = create_aa(aa_layer, channels=width, stride=stride, enable=use_aa)

        self.conv3 = nn.Conv2d(width, outplanes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(outplanes)

        self.se = create_attn(attn_layer, outplanes)

        self.act3 = act_layer(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.drop_path = drop_path

    def zero_init_last(self):
        if getattr(self.bn3, 'weight', None) is not None:
            nn.init.zeros_(self.bn3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop_block(x)
        x = self.act2(x)
        x = self.aa(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.downsample is not None:
            shortcut = self.downsample(shortcut)
        x += shortcut
        x = self.act3(x)

        return x


def downsample_conv(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False),
        norm_layer(out_channels)
    ])


def downsample_avg(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    if stride == 1 and dilation == 1:
        pool = nn.Identity()
    else:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)

    return nn.Sequential(*[
        pool,
        nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
        norm_layer(out_channels)
    ])


def downsample_dwc(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False, groups=in_channels),
    ])


def downsample_dwcn(
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        first_dilation: Optional[int] = None,
        norm_layer: Optional[Type[nn.Module]] = None,
) -> nn.Module:
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    first_dilation = (first_dilation or dilation) if kernel_size > 1 else 1
    p = get_padding(kernel_size, stride, first_dilation)

    return nn.Sequential(*[
        nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=p, dilation=first_dilation, bias=False, groups=in_channels),
        norm_layer(out_channels)
    ])


def drop_blocks(drop_prob: float = 0.):
    return [
        None, None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=1, gamma_scale=0.25) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=1, gamma_scale=0.5) if drop_prob else None,
        partial(DropBlock2d, drop_prob=drop_prob, block_size=1, gamma_scale=1.00) if drop_prob else None]


def make_blocks(
        block_fn: Union[BasicBlock, Bottleneck],
        channels: List[int],
        block_repeats: List[int],
        inplanes: int,
        reduce_first: int = 1,
        output_stride: int = 32,
        down_kernel_size: int = 1,
        residual_down: str = 'conv',
        drop_block_rate: float = 0.,
        drop_path_rate: float = 0.,
        **kwargs,
) -> Tuple[List[Tuple[str, nn.Module]], List[Dict[str, Any]]]:
    stages = []
    feature_info = []
    net_num_blocks = sum(block_repeats)
    net_block_idx = 0
    net_stride = 2
    dilation = prev_dilation = 1
    for stage_idx, (planes, num_blocks, db) in enumerate(zip(channels, block_repeats, drop_blocks(drop_block_rate))):
        stage_name = f'layer{stage_idx + 1}'  # never liked this name, but weight compat requires it
        stride = 1 if stage_idx == 0 else 2
        if net_stride >= output_stride:
            dilation *= stride
            stride = 1
        else:
            net_stride *= stride

        downsample = None
        if stride != 1 or inplanes != planes * block_fn.expansion:
            down_kwargs = dict(
                in_channels=inplanes,
                out_channels=planes * block_fn.expansion,
                kernel_size=down_kernel_size,
                stride=stride,
                dilation=dilation,
                first_dilation=prev_dilation,
                norm_layer=kwargs.get('norm_layer'),
            )
            if residual_down == 'avg':
                downsample = downsample_avg(**down_kwargs)
            elif residual_down == 'dwc':
                downsample = downsample_dwc(**down_kwargs)
            elif residual_down == 'dwcn':
                downsample = downsample_dwcn(**down_kwargs)
            elif residual_down == 'conv':
                downsample = downsample_conv(**down_kwargs)

        block_kwargs = dict(reduce_first=reduce_first, dilation=dilation, drop_block=db, **kwargs)
        blocks = []
        for block_idx in range(num_blocks):
            downsample = downsample if block_idx == 0 else None
            stride = stride if block_idx == 0 else 1
            block_dpr = drop_path_rate * net_block_idx / (net_num_blocks - 1)  # stochastic depth linear decay rule
            blocks.append(block_fn(
                inplanes,
                planes,
                stride,
                downsample,
                first_dilation=prev_dilation,
                drop_path=DropPath(block_dpr) if block_dpr > 0. else None,
                **block_kwargs,
            ))
            prev_dilation = dilation
            inplanes = planes * block_fn.expansion
            net_block_idx += 1

        stages.append((stage_name, nn.Sequential(*blocks)))
        feature_info.append(dict(num_chs=inplanes, reduction=net_stride, module=stage_name))

    return stages, feature_info


# https://github.com/EIFY/mup-vit
# Taken from https://github.com/lucidrains/vit-pytorch, likely ported from https://github.com/google-research/big_vision/
def posemb_sincos_2d(h, w, dim, temperature: int = 10000, dtype = torch.float32):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"
    omega = torch.arange(dim // 4) / (dim // 4 - 1)
    omega = 1.0 / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class LearnedPositionalEmbedding1D(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len, emb_dim)`"""
        x = x + self.pos_embedding
        return x


class LRResNet(nn.Module):
    """LRResNet / ResNeXt / SE-ResNeXt / SE-Net

    This class implements all variants of LRResNet, ResNeXt, SE-ResNeXt, and SENet that
      * have > 1 stride in the 3x3 conv layer of bottleneck
      * have conv-bn-act ordering

    This LRResNet impl supports a number of stem and downsample options based on the v1c, v1d, v1e, and v1s
    variants included in the MXNet Gluon LRResNetV1b model. The C and D variants are also discussed in the
    'Bag of Tricks' paper: https://arxiv.org/pdf/1812.01187. The B variant is equivalent to torchvision default.

    LRResNet variants (the same modifications can be used in SE/ResNeXt models as well):
      * normal, b - 7x7 stem, stem_width = 64, same as torchvision LRResNet, NVIDIA LRResNet 'v1.5', Gluon v1b
      * c - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64)
      * d - 3 layer deep 3x3 stem, stem_width = 32 (32, 32, 64), average pool in downsample
      * e - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128), average pool in downsample
      * s - 3 layer deep 3x3 stem, stem_width = 64 (64, 64, 128)
      * t - 3 layer deep 3x3 stem, stem width = 32 (24, 48, 64), average pool in downsample
      * tn - 3 layer deep 3x3 stem, stem width = 32 (24, 32, 64), average pool in downsample

    ResNeXt
      * normal - 7x7 stem, stem_width = 64, standard cardinality and base widths
      * same c,d, e, s variants as LRResNet can be enabled

    SE-ResNeXt
      * normal - 7x7 stem, stem_width = 64
      * same c, d, e, s variants as LRResNet can be enabled

    SENet-154 - 3 layer deep 3x3 stem (same as v1c-v1s), stem_width = 64, cardinality=64,
        reduction by 2 on width of first bottleneck convolution, 3x3 downsample convs after first block
    """

    def __init__(
            self,
            block: Union[BasicBlock, Bottleneck],
            layers: List[int],
            num_classes: int = 1000,
            in_chans: int = 3,
            output_stride: int = 32,
            global_pool: str = 'avg',
            cardinality: int = 1,
            base_width: int = 64,
            stem_width: int = 64,
            stem_type: str = '',
            replace_stem_pool: bool = False,
            block_reduce_first: int = 1,
            down_kernel_size: int = 1,
            residual_down: str = 'conv',
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[Type[nn.Module]] = None,
            drop_rate: float = 0.0,
            drop_path_rate: float = 0.,
            drop_block_rate: float = 0.,
            zero_init_last: bool = True,
            block_args: Optional[Dict[str, Any]] = None,
            transformer_blocks: int = 0,
            mlp_ratio: float = 0.25,
            img_size: int = 224,
            pos_embedding_type: str = 'sin2d',
            drop_cls_token: bool = False,
            init_values: float = 1e-5,
            inter_feats: bool = False,
    ):
        """
        Args:
            block (nn.Module): class for the residual block. Options are BasicBlock, Bottleneck.
            layers (List[int]) : number of layers in each block
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            cardinality (int): number of convolution groups for 3x3 conv in Bottleneck. (default 1)
            base_width (int): bottleneck channels factor. `planes * base_width / 64 * cardinality` (default 64)
            stem_width (int): number of channels in stem convolutions (default 64)
            stem_type (str): The type of stem (default ''):
                * 'patch': 2x2 non-overlapping (vit style, also used in Swin and CNX)
                * 'patch_overlap': 2x2 overlap with stride 1
                * 'patch_dw': 2x2 non-overlapping with dw conv
                * 'patch_dw_overlap': 2x2 with 2x2 stride and padding=2
                * '', default - a single 7x7 conv with a width of stem_width
                * 'deep' - three 3x3 convolution layers of widths stem_width, stem_width, stem_width * 2
                * 'deep_tiered' - three 3x3 conv layers of widths stem_width//4 * 3, stem_width, stem_width * 2
            replace_stem_pool (bool): replace stem max-pooling layer with a 3x3 stride-2 convolution
            block_reduce_first (int): Reduction factor for first convolution output width of residual blocks,
                1 for all archs except senets, where 2 (default 1)
            down_kernel_size (int): kernel size of residual block downsample path,
                1x1 for most, 3x3 for senets (default: 1)
            avg_down (bool): use avg pooling for projection skip connection between stages/downsample (default False) ->
            changed to residual_down
            residual_down (str): use avg pooling for projection skip connection between stages/downsample (default conv)
                * avg: avg + conv
                * conv: conv with stride
                * dwc: dw conv
                * dwcn: dw conv with norm
            act_layer (str, nn.Module): activation layer
            norm_layer (str, nn.Module): normalization layer
            aa_layer (nn.Module): anti-aliasing layer
            drop_rate (float): Dropout probability before classifier, for training (default 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default 0.)
            drop_block_rate (float): Drop block rate (default 0.)
            zero_init_last (bool): zero-init the last weight in residual path (usually last BN affine weight)
            block_args (dict): Extra kwargs to pass through to block module
        """
        super(LRResNet, self).__init__()
        block_args = block_args or dict()
        assert output_stride in (8, 16, 32)
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        
        act_layer = get_act_layer(act_layer)
        norm_layer = get_norm_layer(norm_layer)

        # Stem
        FIRST_CONV_FEATURES = 32
        deep_stem = 'deep' in stem_type
        inplanes = stem_width * 2 if deep_stem else FIRST_CONV_FEATURES

        if 'patch' in stem_type or 'dw' in stem_type:
            ps = 4 if 'overlap' in stem_type else 2
            padding = 1 if 'overlap' in stem_type else 0
            # equivalent flops to traditional patch conv
            dw_chans = int((inplanes * (ps ** 2)) / (in_chans * (ps ** 2) + inplanes))

            if stem_type in ('patch', 'patch_overlap'):
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_chans, inplanes, kernel_size=ps, stride=2, padding=padding, bias=False),
                    norm_layer(inplanes),
                    act_layer(inplace=True),
                )
            elif stem_type in ('patch_dw', 'patch_dw_overlap'):
                self.conv1 = nn.Sequential(
                    nn.Conv2d(in_chans, in_chans * dw_chans, kernel_size=ps, stride=2, padding=padding, bias=False, groups=in_chans),
                    norm_layer(in_chans * dw_chans),
                    act_layer(inplace=True),
                    nn.Conv2d(in_chans * dw_chans, inplanes, kernel_size=1, stride=1, padding=0),
                    norm_layer(inplanes),
                    act_layer(inplace=True),
                )

            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        else:
            if deep_stem:
                stem_chs = (stem_width, stem_width)
                if 'tiered' in stem_type:
                    stem_chs = (3 * (stem_width // 4), stem_width)
                self.conv1 = nn.Sequential(*[
                    nn.Conv2d(in_chans, stem_chs[0], 3, stride=2, padding=1, bias=False),
                    norm_layer(stem_chs[0]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[0], stem_chs[1], 3, stride=1, padding=1, bias=False),
                    norm_layer(stem_chs[1]),
                    act_layer(inplace=True),
                    nn.Conv2d(stem_chs[1], inplanes, 3, stride=1, padding=1, bias=False)])
            else:
                self.conv1 = nn.Conv2d(in_chans, inplanes, kernel_size=3, stride=2, padding=1, bias=False)
            self.bn1 = norm_layer(inplanes)
            self.act1 = act_layer(inplace=True)
            self.feature_info = [dict(num_chs=inplanes, reduction=2, module='act1')]

        # Feature Blocks
        OUTPUT_FEATURES = 512
        channels = [32, 64, 128, 256, OUTPUT_FEATURES]
        stage_modules, stage_feature_info = make_blocks(
            block,
            channels,
            layers,
            inplanes,
            cardinality=cardinality,
            base_width=base_width,
            output_stride=output_stride,
            reduce_first=block_reduce_first,
            residual_down=residual_down,
            down_kernel_size=down_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            drop_block_rate=drop_block_rate,
            drop_path_rate=drop_path_rate,
            **block_args,
        )
        for stage in stage_modules:
            self.add_module(*stage)  # layer1, layer2, etc
        self.feature_info.extend(stage_feature_info)

        self.num_features = OUTPUT_FEATURES * block.expansion

        # transformer blocks
        if transformer_blocks:
            fh = img_size // output_stride
            seq_len = int(fh ** 2)

            if inter_feats:
                self.inter_pool = nn.AdaptiveAvgPool2d(fh)
                num_features = sum([c * block.expansion for c in channels]) + inplanes
            else:
                num_features = self.num_features

            self.cls_token = nn.Parameter(torch.zeros(1, 1, num_features))

            # Positional embedding
            if pos_embedding_type == 'learned':
                self.positional_embedding = LearnedPositionalEmbedding1D(
                    seq_len, num_features)
            elif pos_embedding_type == 'sin2d':
                self.register_buffer("sin2d_embedding", posemb_sincos_2d(fh, fh, num_features))

            # Transformer encoder
            num_heads = num_features // 64
            assert int(num_heads * 64) == num_features, 'num_features need to be divisible by 64'
            self.encoder = nn.ModuleList([TransformerParallelScalingBlock(
                num_features,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                init_values=init_values,
                drop_path=drop_path_rate,
            ) for i in range(transformer_blocks)])

            self.encoder_norm = nn.LayerNorm(num_features, eps=1e-6)

            self.global_pool = nn.Identity()
            self.fc = nn.Identity()

            if self.num_classes:
                self.classifier_cls = True
                self.fc = nn.Linear(num_features, self.num_classes)

            if drop_cls_token:
                self.drop_class_token = True

        else:
            # Head (Pooling and Classifier)
            self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

        self.init_weights(zero_init_last=zero_init_last, residual_down=residual_down)

    @torch.jit.ignore
    def init_weights(self, zero_init_last: bool = True, residual_down: str = 'conv'):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'dwc' in residual_down and 'downsample' in n:
                print('Initialized to near one values: ', n)
                nn.init.normal_(m.weight, mean=1.0, std=0.001)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if zero_init_last:
            for m in self.modules():
                if hasattr(m, 'zero_init_last'):
                    m.zero_init_last()
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=1e-6)
        if hasattr(self, 'positional_embedding'):
            trunc_normal_(self.positional_embedding.pos_embedding, std=.02)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        matcher = dict(stem=r'^conv1|bn1|maxpool', blocks=r'^layer(\d+)' if coarse else r'^layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self, name_only: bool = False):
        return 'fc' if name_only else self.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.global_pool, self.fc = create_classifier(self.num_features, self.num_classes, pool_type=global_pool)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'inter_pool'):
            inter_feats = []

        b = x.shape[0]
        x = self.conv1(x)
        if hasattr(self, 'inter_pool'):
            inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))


        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq([self.layer1, self.layer2, self.layer3, self.layer4, self.layer5], x, flatten=True)
        else:
            x = self.layer1(x)
            if hasattr(self, 'inter_pool'):
                inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))

            x = self.layer2(x)
            if hasattr(self, 'inter_pool'):
                inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))

            x = self.layer3(x)
            if hasattr(self, 'inter_pool'):
                inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))

            x = self.layer4(x)
            if hasattr(self, 'inter_pool'):
                inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))

            x = self.layer5(x)
            if hasattr(self, 'inter_pool'):
                inter_feats.append(rearrange(self.inter_pool(x), 'b d fh fw -> b (fh fw) d'))


        if hasattr(self, 'encoder'):
            if hasattr(self, 'inter_pool'):
                x = torch.cat(inter_feats, dim=-1)
            else:
                x = rearrange(x, 'b d fh fw -> b (fh fw) d')

            if hasattr(self, 'sin2d_embedding'):
                x = x + self.sin2d_embedding

            if hasattr(self, 'cls_token'):
                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
                x = torch.cat((cls_tokens, x), dim=1)

            if hasattr(self, 'positional_embedding'):
                x = self.positional_embedding(x)

            for i, blk in enumerate(self.encoder):
                x = blk(x)

            x = self.encoder_norm(x)

        if hasattr(self, 'drop_class_token'):
            x = x[:, 1:]

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if hasattr(self, 'encoder') and hasattr(self, 'classifier_cls'):
            return self.fc(x[:, 0])

        x = self.global_pool(x)
        if self.drop_rate:
            x = F.dropout(x, p=float(self.drop_rate), training=self.training)
        return x if pre_logits else self.fc(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _create_lrresnet(variant, pretrained: bool = False, **kwargs) -> LRResNet:
    return build_model_with_cfg(LRResNet, variant, pretrained, **kwargs)


@register_model
def lrresnet22(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2])
    return _create_lrresnet('lrresnet22', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_width=16,
                      stem_type='deep', residual_down='avg')
    return _create_lrresnet('lrresnet22d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet22dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet18t2dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-T-2-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1],
                      residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet18t2dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet18t2d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-T-2-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_width=16,
                      stem_type='deep', residual_down='avg', transformer_blocks=2)
    return _create_lrresnet('lrresnet18t2d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet18t2ddwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-T-2-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_width=16,
                      stem_type='deep', residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet18t2ddwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet18t2pdwosrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-T-2 DW Overlap Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='patch_dw_overlap',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet18t2pdwosrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet18t2pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-T-2 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet18t2pdwsrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet18t4pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-18-T-4 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 1, 1], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=4)
    return _create_lrresnet('lrresnet18t4pdwsrnk', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet34
@register_model
def lrresnet38(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-38 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 6, 3])
    return _create_lrresnet('lrresnet38', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet38d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-38-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 6, 3], stem_width=16,
                      stem_type='deep', residual_down='avg')
    return _create_lrresnet('lrresnet38d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet38dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-38 DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 6, 5], residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet38dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet34t2dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-T-2 model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2],
                      residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet34t2dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet34t2d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-T-2-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_width=16,
                      stem_type='deep', residual_down='avg', transformer_blocks=2)
    return _create_lrresnet('lrresnet34t2d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet34t2ddwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-T-2-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_width=16,
                      stem_type='deep', residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet34t2ddwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet34t2pdwosrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-T-2 DW Overlap Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='patch_dw_overlap',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet34t2pdwosrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet34t2pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-T-2 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet34t2pdwswrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet34t4pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-34-T-4 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 5, 2], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=4)
    return _create_lrresnet('lrresnet34t4pdwswrnk', pretrained, **dict(model_args, **kwargs))


# equivalent to resnet50
@register_model
def lrresnet56(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-56 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 6, 3])
    return _create_lrresnet('lrresnet56', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet56d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-56-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 6, 3], stem_width=16,
                      stem_type='deep', residual_down='avg')
    return _create_lrresnet('lrresnet56d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet56dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-56 DW Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 6, 3], residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet56dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet50t2dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-T-2-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2],
                      residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet50t2dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet50t2d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-T-2-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_width=16,
                      stem_type='deep', residual_down='avg', transformer_blocks=2)
    return _create_lrresnet('lrresnet50t2d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet50t2ddwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-T-2-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_width=16,
                      stem_type='deep', residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet50t2ddwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet50t2pdwosrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-T-2 DW Overlap Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='patch_dw_overlap',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet50t2pdwosrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet50t2pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-T-2 DW Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet50t2pdwsrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet50t4pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-50-T-4 DW Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 5, 2], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=4)
    return _create_lrresnet('lrresnet50t4pdwsrnk', pretrained, **dict(model_args, **kwargs))

# equivalent to resnet101
@register_model
def lrresnet107(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-107 model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 23, 3])
    return _create_lrresnet('lrresnet107', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet107d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-107-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 23, 3], stem_width=16,
                      stem_type='deep', residual_down='avg')
    return _create_lrresnet('lrresnet107d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet107dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101 DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 3, 4, 23, 3], residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet107', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet101t2dwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-T-2-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2],
                      residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet101t2dwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet101t2d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-T-2-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_width=16,
                      stem_type='deep', residual_down='avg', transformer_blocks=2)
    return _create_lrresnet('lrresnet101t2d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet101t2ddwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-T-2-D model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_width=16,
                      stem_type='deep', residual_down='dwc', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet101t2ddwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet101t2pdwosrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-T-2 DW Overlap Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='patch_dw_overlap',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet101t2pdwosrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet101t2pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-T-2 DW Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2)
    return _create_lrresnet('lrresnet101t2pdwsrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet101t4pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-101-T-4 DW Stem + Residual model.
    """
    model_args = dict(block=Bottleneck, layers=[2, 3, 4, 22, 2], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=4)
    return _create_lrresnet('lrresnet101t4pdwsrnk', pretrained, **dict(model_args, **kwargs))



# other variants
@register_model
def lrresnet22pdwosrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw_overlap', residual_down='dwcn', down_kernel_size=3)
    return _create_lrresnet('lrresnet22pdwosrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdwosrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw_overlap', residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet22pdwosrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdwosr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw_overlap', residual_down='dwc')
    return _create_lrresnet('lrresnet22pdwosr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdwos(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw_overlap')
    return _create_lrresnet('lrresnet22pdwos', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw', residual_down='dwcn', down_kernel_size=3)
    return _create_lrresnet('lrresnet22pwdwsrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdwsrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw', residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet22pdwsrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdwsr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw', residual_down='dwc')
    return _create_lrresnet('lrresnet22pdwsr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22pdws(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Stem model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_type='patch_dw')
    return _create_lrresnet('lrresnet22pwdws', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22dwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], residual_down='dwc')
    return _create_lrresnet('lrresnet22dwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22dwrn(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], residual_down='dwcn')
    return _create_lrresnet('lrresnet22dwrn', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22dwrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], residual_down='dwcn', down_kernel_size=3)
    return _create_lrresnet('lrresnet22dwrnk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22crk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 Conv Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], residual_down='conv', down_kernel_size=3)
    return _create_lrresnet('lrresnet22crk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22ark(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22 Conv Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], residual_down='avg', down_kernel_size=3)
    return _create_lrresnet('lrresnet22ark', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22ddwr(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22-D DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_width=16, stem_type='deep', residual_down='dwc')
    return _create_lrresnet('lrresnet22ddwr', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet22ddwrk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-22-D DW Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 2, 2, 2], stem_width=16, stem_type='deep', residual_down='dwc', down_kernel_size=3)
    return _create_lrresnet('lrresnet22ddwrk', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet16ift2d(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-16-IFT-2-D model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 1, 1, 1], stem_width=16,
                      stem_type='deep', residual_down='avg', transformer_blocks=2,
                      inter_feats=True)
    return _create_lrresnet('lrresnet16ift2d', pretrained, **dict(model_args, **kwargs))


@register_model
def lrresnet16ift2pdwsrnk(pretrained: bool = False, **kwargs) -> LRResNet:
    """Constructs a LRResNet-16-IFT-2 DW Stem + Residual model.
    """
    model_args = dict(block=BasicBlock, layers=[2, 2, 1, 1, 1], stem_type='patch_dw',
                      residual_down='dwcn', down_kernel_size=3, transformer_blocks=2,
                      inter_feats=True)
    return _create_lrresnet('lrresnet16ift2pdwsrnk', pretrained, **dict(model_args, **kwargs))
