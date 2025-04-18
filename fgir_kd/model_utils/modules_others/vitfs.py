""" Vision Transformer (ViT) in PyTorch

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
import random
import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final
from einops.layers.torch import Rearrange
from einops import repeat, reduce, rearrange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import named_apply, checkpoint_seq, adapt_input_conv
from timm.models._registry import generate_default_cfgs, register_model


__all__ = ['ViTFS']  # model_registry will add each entrypoint fn to this


_logger = logging.getLogger(__name__)


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


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size=3, stride=2, padding=1, activation='relu', norm='batchnorm'):
        super(ConvLayer, self).__init__()

        if in_channels == out_channels:
            stride = 1
            
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
            kernel_size=(kernel_size, kernel_size), stride=(stride, stride), padding=(padding, padding))
        if activation == 'relu':
            self.activation = nn.ReLU()
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x) 
        if hasattr(self, 'activation'):
            x = self.activation(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        return x


class ConvPatchingStem(nn.Module):
    def __init__(self, img_size=224, input_channels=3, hidden_size=768, patch_size=16):
        super(ConvPatchingStem, self).__init__()

        inc = input_channels

        if patch_size == 16:
            if hidden_size == 768:
                channels_in_list = [inc, 64, 128, 128, 256, 256]
                channels_out_list = [64, 128, 128, 256, 256, 512]
            elif hidden_size == 384:
                channels_in_list = [inc, 48, 96, 192]
                channels_out_list = [48, 96, 192, 384]
            elif hidden_size == 192:
                channels_in_list = [inc, 24, 48, 96]
                channels_out_list = [24, 48, 96, 192]
            elif hidden_size == 96:
                channels_in_list = [inc, 12, 24, 48]
                channels_out_list = [12, 24, 48, 96]
            elif hidden_size == 48:
                channels_in_list = [inc, 6, 12, 24]
                channels_out_list = [6, 12, 24, 48]
        elif patch_size == 8:
            if hidden_size == 768:
                channels_in_list = [inc, 128, 128, 256]
                channels_out_list = [128, 128, 256, 512]
            elif hidden_size == 384:
                channels_in_list = [inc, 96, 192]
                channels_out_list = [96, 192, 384]
            elif hidden_size == 192:
                channels_in_list = [inc, 48, 96]
                channels_out_list = [48, 96, 192]
            elif hidden_size == 96:
                channels_in_list = [inc, 24, 48]
                channels_out_list = [24, 48, 96]
            elif hidden_size == 48:
                channels_in_list = [inc, 12, 24]
                channels_out_list = [12, 24, 48]


        self.conv3x3layers = nn.ModuleList([
            ConvLayer(channels_in, channels_out)
            for (channels_in, channels_out) in zip(channels_in_list, channels_out_list)
        ])


        self.conv1x1 = nn.Conv2d(channels_out_list[-1], hidden_size, kernel_size=1,
                                 stride=1, padding=0, bias=False)

        self.num_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        print(x.shape)
        for layer in self.conv3x3layers: 
            x = layer(x)
            print(x.shape)

        x = self.conv1x1(x)

        x = rearrange(x, 'b d fh fw -> b (fh fw) d')
        # print(x.shape)

        return x


class DWConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, 
    kernel_size=3, stride=2, padding=1, activation='relu6', norm='batchnorm',
    ):
        super(DWConvLayer, self).__init__()

        channels_per_group = out_channels // in_channels

        if in_channels == out_channels:
            stride = 1
            
        self.dwconv = nn.Conv2d(
            in_channels=in_channels, out_channels=in_channels * channels_per_group,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            groups=in_channels,
        )

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'relu6':
            self.activation = nn.ReLU6()
        if norm == 'batchnorm':
            self.norm = nn.BatchNorm2d(in_channels * channels_per_group)
            self.norm2 = nn.BatchNorm2d(out_channels)

        self.pwconv = nn.Conv2d(in_channels * channels_per_group, out_channels, 1, 1, 0)

        if stride != 1:
            # residual preserving downsampling
            self.rpds = nn.Conv2d(
                in_channels=in_channels, out_channels=in_channels * channels_per_group,
                kernel_size=1,
                stride=(stride, stride),
                padding=0,
                groups=in_channels,
            )

    def forward(self, x):
        if hasattr(self, 'rpds'):
            res = x

        x = self.dwconv(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)

        x = self.pwconv(x)
        if hasattr(self, 'activation'):
            x = self.activation(x)
        if hasattr(self, 'norm2'):
            x = self.norm2(x)        

        if hasattr(self, 'rpds'):
            x = x + self.rpds(res)

        return x


class DWConvPatchingStem(nn.Module):
    def __init__(self, img_size=224, input_channels=3, hidden_size=768, patch_size=16):
        super(DWConvPatchingStem, self).__init__()

        inc = input_channels

        if patch_size == 16:
            if hidden_size == 768:
                channels_in_list = [inc, 64, 128, 128, 256, 256]
                channels_out_list = [64, 128, 128, 256, 256, 512]
            elif hidden_size == 384:
                channels_in_list = [inc, 48, 96, 192]
                channels_out_list = [48, 96, 192, 384]
            elif hidden_size == 192:
                channels_in_list = [inc, 24, 48, 96]
                channels_out_list = [24, 48, 96, 192]
            elif hidden_size == 96:
                channels_in_list = [inc, 12, 24, 48]
                channels_out_list = [12, 24, 48, 96]
            elif hidden_size == 48:
                channels_in_list = [inc, 6, 12, 24]
                channels_out_list = [6, 12, 24, 48]
        elif patch_size == 8:
            if hidden_size == 768:
                channels_in_list = [inc, 128, 128, 256]
                channels_out_list = [128, 128, 256, 512]
            elif hidden_size == 384:
                channels_in_list = [inc, 96, 192]
                channels_out_list = [96, 192, 384]
            elif hidden_size == 192:
                channels_in_list = [inc, 48, 96]
                channels_out_list = [48, 96, 192]
            elif hidden_size == 96:
                channels_in_list = [inc, 24, 48]
                channels_out_list = [24, 48, 96]
            elif hidden_size == 48:
                channels_in_list = [inc, 12, 24]
                channels_out_list = [12, 24, 48]


        self.conv3x3layers = nn.ModuleList([
            DWConvLayer(channels_in, channels_out)
            for (channels_in, channels_out) in zip(channels_in_list, channels_out_list)
        ])
        print(self.conv3x3layers)

        self.conv1x1 = nn.Conv2d(channels_out_list[-1], hidden_size, kernel_size=1,
                                 stride=1, padding=0, bias=False)

        # residual preserving downsampling
        self.rpds = nn.Conv2d(
            in_channels=input_channels, out_channels=channels_out_list[-1],
            kernel_size=1,
            # kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            padding=0,
            groups=input_channels,
        )

        self.num_patches = (img_size // patch_size) ** 2


    @torch.jit.ignore
    def init_weights(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and 'rpds' in n:
                print('Initialized to near one values: ', n)
                nn.init.normal_(m.weight, mean=1.0, std=0.001)
                nn.init.normal_(m.bias, mean=0.0, std=0.001)


    def forward(self, x):
        # print(x.shape)

        if hasattr(self, 'rpds'):
            res = x

        for layer in self.conv3x3layers: 
            x = layer(x)
            # print(x.shape)

        x = self.conv1x1(x)

        if hasattr(self, 'rpds'):
            x = x + self.rpds(res)

        x = rearrange(x, 'b d fh fw -> b (fh fw) d')
        # print(x.shape)

        return x


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
        self.fused_attn = False

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.pre_softmax = nn.Identity()
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
            attn = self.pre_softmax(attn)
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


class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
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
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x



class ViTFS(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """
    dynamic_img_size: Final[bool]

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patchifier: str = 'convstem',
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            global_pool: Literal['', 'avg', 'token', 'map'] = 'token',
            embed_dim: int = 768,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.,
            qkv_bias: bool = True,
            qk_norm: bool = False,
            init_values: Optional[float] = None,
            class_token: bool = True,
            sin2d_embedding: bool = True,
            no_embed_class: bool = False,
            reg_tokens: int = 0,
            pre_norm: bool = False,
            fc_norm: Optional[bool] = None,
            dynamic_img_size: bool = False,
            dynamic_img_pad: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.0,
            weight_init: Literal['skip', 'jax', 'jax_nlhb', 'moco', ''] = '',
            embed_layer: Callable = PatchEmbed,
            norm_layer: Optional[LayerType] = None,
            act_layer: Optional[LayerType] = None,
            block_fn: Type[nn.Module] = Block,
            mlp_layer: Type[nn.Module] = Mlp,
            args=None,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token', 'map')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = get_norm_layer(norm_layer) or partial(nn.LayerNorm, eps=1e-6)
        act_layer = get_act_layer(act_layer) or nn.GELU

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt='NHWC'))

        if patchifier == 'convstem':
            self.patch_embed = ConvPatchingStem(img_size, in_chans, embed_dim, patch_size)
        elif patchifier == 'dwconvstem':
            self.patch_embed = DWConvPatchingStem(img_size, in_chans, embed_dim, patch_size)
        else:
            self.patch_embed = embed_layer(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
                dynamic_img_pad=dynamic_img_pad,
                **embed_args,
            )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = nn.Parameter(torch.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None

        if sin2d_embedding:
            fh = int(num_patches ** 0.5)
            self.register_buffer("sin2d_embedding", posemb_sincos_2d(fh, fh, embed_dim))
        else:
            embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer(embed_dim) if pre_norm else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                mlp_layer=mlp_layer,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        if global_pool == 'map':
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.args = args
        self.debugging = getattr(args, 'debugging', False)

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode: Literal['jax', 'jax_nlhb', 'moco', ''] = '') -> None:
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        if hasattr(self, 'pos_embed'):
            trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m: nn.Module) -> None:
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path: str, prefix: str = '') -> None:
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self) -> Set:
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool = None) -> None:
        self.num_classes = num_classes

        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token', 'map')
            if global_pool == 'map' and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != 'map ' and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool

        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        return 0

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'sin2d_embedding'):
            x = x + self.sin2d_embedding
        elif self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if hasattr(self, 'sin2d_embedding'):
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        elif self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        elif not hasattr(self, 'sin2d_embedding'):
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
    ) -> List[torch.Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
            self,
            x: torch.Tensor,
            n: Union[int, Sequence] = 1,
            reshape: bool = False,
            return_prefix_tokens: bool = False,
            norm: bool = False,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        """ Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0:self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens:] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def get_new_module_names(self):
        return []

    def maybe_print(self, *args):
        if self.debugging:
            print(*args)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # feature extraction
        x = self.patch_embed(x)
        b, s, d = x.shape
        self.maybe_print('Patchifier output: ', x.shape)

        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        for i, blck in enumerate(self.blocks):
            x = blck(x)
            self.maybe_print('Encoder blocks: ', i, x.shape)

        if self.reg_token is not None and self.cls_token is not None:
            x = torch.cat([x[:, :1], x[:, -s:]], dim=1)
        elif self.reg_token is not None and self.cls_token is None:
            x = torch.cat([x[:, :s]], dim=1)

        x = self.norm(x)
        self.maybe_print('Encoder norm output: ', x.shape)

        return x

    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        # pooling
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == 'avg':
            x = x[:, self.num_prefix_tokens:].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        return x

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        # classificationi head
        x = self.fc_norm(x)
        x = self.head_drop(x)
        x = self.head(x)
        return x


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extract_features(x)
        x = self.pool_features(x)
        x = self.classify(x)
        return x

def init_weights_vit_timm(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.0) -> None:
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_moco(module: nn.Module, name: str = '') -> None:
    """ ViT weight initialization, matching moco-v3 impl minus fixed PatchEmbed """
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode: str = 'jax', head_bias: float = 0.0) -> None:
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.

    *DEPRECATED* This function is being deprecated in favour of resample_abs_pos_embed

    Adapted from:
        https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    """
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
        ntok_new -= num_prefix_tokens
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info(f'Resized position embedding: {posemb.shape} ({[gs_old, gs_old]}) to {posemb_new.shape} ({gs_new}).')
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=interpolation, antialias=antialias, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


@torch.no_grad()
def _load_weights(model: ViTFS, checkpoint_path: str, prefix: str = '') -> None:
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    interpolation = 'bilinear'
    antialias = False
    big_vision = False
    if not prefix:
        if 'opt/target/embedding/kernel' in w:
            prefix = 'opt/target/'
        elif 'params/embedding/kernel' in w:
            prefix = 'params/'
            big_vision = True
        elif 'params/img/embedding/kernel' in w:
            prefix = 'params/img/'
            big_vision = True

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    if embed_conv_w.shape[-2:] != model.patch_embed.proj.weight.shape[-2:]:
        embed_conv_w = resample_patch_embed(
            embed_conv_w,
            model.patch_embed.proj.weight.shape[-2:],
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )

    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    if model.cls_token is not None:
        model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    if big_vision:
        pos_embed_w = _n2p(w[f'{prefix}pos_embedding'], t=False)
    else:
        pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        old_shape = pos_embed_w.shape
        num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
        pos_embed_w = resample_abs_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w,
            new_size=model.patch_embed.grid_size,
            num_prefix_tokens=num_prefix_tokens,
            interpolation=interpolation,
            antialias=antialias,
            verbose=True,
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if (isinstance(model.head, nn.Linear) and
            f'{prefix}head/bias' in w and
            model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]):
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))
    # NOTE representation layer has been removed, not used in latest 21k/1k pretrained weights
    # if isinstance(getattr(model.pre_logits, 'fc', None), nn.Linear) and f'{prefix}pre_logits/bias' in w:
    #     model.pre_logits.fc.weight.copy_(_n2p(w[f'{prefix}pre_logits/kernel']))
    #     model.pre_logits.fc.bias.copy_(_n2p(w[f'{prefix}pre_logits/bias']))
    if model.attn_pool is not None:
        block_prefix = f'{prefix}MAPHead_0/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_0/'
        model.attn_pool.latent.copy_(_n2p(w[f'{block_prefix}probe'], t=False))
        model.attn_pool.kv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('key', 'value')]))
        model.attn_pool.kv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('key', 'value')]))
        model.attn_pool.q.weight.copy_(_n2p(w[f'{mha_prefix}query/kernel'], t=False).flatten(1).T)
        model.attn_pool.q.bias.copy_(_n2p(w[f'{mha_prefix}query/bias'], t=False).reshape(-1))
        model.attn_pool.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        model.attn_pool.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        model.attn_pool.norm.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        model.attn_pool.norm.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        for r in range(2):
            getattr(model.attn_pool.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/kernel']))
            getattr(model.attn_pool.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_0/Dense_{r}/bias']))

    mha_sub, b_sub, ln1_sub = (0, 0, 1) if big_vision else (1, 3, 2)
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + f'MultiHeadDotProductAttention_{mha_sub}/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_{ln1_sub}/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_{b_sub}/Dense_{r}/bias']))


def _convert_openai_clip(
        state_dict: Dict[str, torch.Tensor],
        model: ViTFS,
        prefix: str = 'visual.',
) -> Dict[str, torch.Tensor]:
    out_dict = {}
    swaps = [
        ('conv1', 'patch_embed.proj'),
        ('positional_embedding', 'pos_embed'),
        ('transformer.resblocks.', 'blocks.'),
        ('ln_pre', 'norm_pre'),
        ('ln_post', 'norm'),
        ('ln_', 'norm'),
        ('in_proj_', 'qkv.'),
        ('out_proj', 'proj'),
        ('mlp.c_fc', 'mlp.fc1'),
        ('mlp.c_proj', 'mlp.fc2'),
    ]
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        k = k.replace(prefix, '')
        for sp in swaps:
            k = k.replace(sp[0], sp[1])

        if k == 'proj':
            k = 'head.weight'
            v = v.transpose(0, 1)
            out_dict['head.bias'] = torch.zeros(v.shape[0])
        elif k == 'class_embedding':
            k = 'cls_token'
            v = v.unsqueeze(0).unsqueeze(1)
        elif k == 'pos_embed':
            v = v.unsqueeze(0)
            if v.shape[1] != model.pos_embed.shape[1]:
                # To resize pos embedding when using model at different size from pretrained weights
                v = resize_pos_embed(
                    v,
                    model.pos_embed,
                    0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                    model.patch_embed.grid_size
                )
        out_dict[k] = v
    return out_dict


def _convert_dinov2(
        state_dict: Dict[str, torch.Tensor],
        model: ViTFS,
) -> Dict[str, torch.Tensor]:
    import re
    out_dict = {}
    state_dict.pop("mask_token", None)
    if 'register_tokens' in state_dict:
        # convert dinov2 w/ registers to no_embed_class timm model (neither cls or reg tokens overlap pos embed)
        out_dict['reg_token'] = state_dict.pop('register_tokens')
        out_dict['cls_token'] = state_dict.pop('cls_token') + state_dict['pos_embed'][:, 0]
        out_dict['pos_embed'] = state_dict.pop('pos_embed')[:, 1:]
    for k, v in state_dict.items():
        if re.match(r"blocks\.(\d+)\.mlp\.w12\.(?:weight|bias)", k):
            out_dict[k.replace("w12", "fc1")] = v
            continue
        elif re.match(r"blocks\.(\d+)\.mlp\.w3\.(?:weight|bias)", k):
            out_dict[k.replace("w3", "fc2")] = v
            continue
        out_dict[k] = v
    return out_dict


def checkpoint_filter_fn(
        state_dict: Dict[str, torch.Tensor],
        model: ViTFS,
        adapt_layer_scale: bool = False,
        interpolation: str = 'bicubic',
        antialias: bool = True,
) -> Dict[str, torch.Tensor]:
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    prefix = ''

    if 'visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model)
    elif 'module.visual.class_embedding' in state_dict:
        return _convert_openai_clip(state_dict, model, prefix='module.visual.')

    if "mask_token" in state_dict:
        state_dict = _convert_dinov2(state_dict, model)

    if "encoder" in state_dict:
        state_dict = state_dict['encoder']
        prefix = 'module.'

    if 'module.cls_token' in state_dict:
        # mocov3
        prefix = 'module.'

    if 'visual.trunk.pos_embed' in state_dict:
        # convert an OpenCLIP model with timm vision encoder
        # FIXME remap final nn.Linear if it exists outside of the timm .trunk (ie in visual.head.proj)
        prefix = 'visual.trunk.'

    if prefix:
        # filter on & remove prefix string from keys
        state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            O, I, H, W = model.patch_embed.proj.weight.shape
            if len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = model.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v,
                    (H, W),
                    interpolation=interpolation,
                    antialias=antialias,
                    verbose=True,
                )
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            # To resize pos embedding when using model at different size from pretrained weights
            num_prefix_tokens = 0 if getattr(model, 'no_embed_class', False) else getattr(model, 'num_prefix_tokens', 1)
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=num_prefix_tokens,
                interpolation=interpolation,
                antialias=antialias,
                verbose=True,
            )
        elif adapt_layer_scale and 'gamma_' in k:
            # remap layer-scale gamma into sub-module (deit3 models)
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            # NOTE representation layer removed as not used in latest 21k/1k pretrained weights
            continue
        out_dict[k] = v
    return out_dict


def _cfg(url: str = '', **kwargs) -> Dict[str, Any]:
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': 0.9,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN,
        'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj',
        'classifier': 'head',
        **kwargs,
    }

default_cfgs = {
    # patch models, imagenet21k (weights from official Google JAX impl), classifier not valid
    # 'vitfs_base_patch16_224.orig_in21k': _cfg(
    #     # url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_patch16_224_in21k-e5005f0a.pth',
    #     hf_hub_id='timm/vit_base_patch16_224.orig_in21k',
    #     num_classes=0),
}
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer(variant: str, pretrained: bool = False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    if 'flexi' in variant:
        # FIXME Google FlexiViT pretrained models have a strong preference for bilinear patch / embed
        # interpolation, other pretrained models resize better w/ anti-aliased bicubic interpolation.
        _filter_fn = partial(checkpoint_filter_fn, interpolation='bilinear', antialias=False)
    else:
        _filter_fn = checkpoint_filter_fn

    # FIXME attn pool (currently only in siglip) params removed if pool disabled, is there a better soln?
    strict = False
    if 'siglip' in variant and kwargs.get('global_pool', None) != 'map':
        strict = False

    return build_model_with_cfg(
        ViTFS,
        variant,
        pretrained,
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=strict,
        **kwargs,
    )


@register_model
def vitfs_base_patch16_224(pretrained: bool = False, **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
    model = _create_vision_transformer('vitfs_base_patch16_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_tiny_patch16_gap_224(pretrained: bool = False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    model_args = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3,
                      class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer('vitfs_tiny_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_small_patch16_gap_224(pretrained: bool = False, **kwargs):
    """ ViT-Small (ViT-S/16)
    """
    model_args = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6,
                      class_token=False, global_pool='avg', fc_norm=False)
    model = _create_vision_transformer('vitfs_small_patch16_gap_224', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_tiny_patch16_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-T/16 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch16_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_small_patch16_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-S/16 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch16_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_base_patch16_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-B/16 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False
    )
    model = _create_vision_transformer(
        'vitfs_base_patch16_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_tiny_patch8_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-T/8 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=8, embed_dim=192, depth=12, num_heads=3, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch8_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

# depthwise stem with residual preserving downsampling rpds

@register_model
def vitfs_tiny_patch16_dwstem_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-T/16 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False,
        patchifier='dwconvstem',
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch16_dwstem_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_tiny_patch8_dwstem_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-T/8 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=8, embed_dim=192, depth=12, num_heads=3, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False,
        patchifier='dwconvstem',
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch8_dwstem_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


# experimental models (scale channels, consider scaling layers instead)

@register_model
def vitfs_nano_patch16_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-n/16 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=16, embed_dim=48, depth=12, num_heads=3, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch16_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model


@register_model
def vitfs_micro_patch16_gap_reg4_dinov2(pretrained: bool = False, **kwargs):
    """ ViT-u/16 for DINOv2 w/ 4 registers
    """
    model_args = dict(
        patch_size=16, embed_dim=96, depth=12, num_heads=3, init_values=1e-5,
        reg_tokens=4, no_embed_class=True, class_token=False, global_pool='avg', fc_norm=False
    )
    model = _create_vision_transformer(
        'vitfs_tiny_patch16_gap_reg4_dinov2', pretrained=pretrained, **dict(model_args, **kwargs))
    return model

