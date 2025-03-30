"""RegNet X, Y, Z, and more

Paper: `Designing Network Design Spaces` - https://arxiv.org/abs/2003.13678
Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py

Paper: `Fast and Accurate Model Scaling` - https://arxiv.org/abs/2103.06877
Original Impl: None

Based on original PyTorch impl linked above, but re-wrote to use my own blocks (adapted from ResNet here)
and cleaned up with more descriptive variable names.

Weights from original pycls impl have been modified:
* first layer from BGR -> RGB as most PyTorch models are
* removed training specific dict entries from checkpoints and keep model state_dict only
* remap names to match the ones here

Supports weight loading from torchvision and classy-vision (incl VISSL SEER)

A number of custom timm model definitions additions including:
* stochastic depth, gradient checkpointing, layer-decay, configurable dilation
* a pre-activation 'V' variant
* only known RegNet-Z model definitions with pretrained weights

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import ClassifierHead, AvgPool2dSame, ConvNormAct, SEModule, DropPath, GroupNormAct
from timm.layers import get_act_layer, get_norm_act_layer, create_conv2d, make_divisible
# from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

__all__ = ['RegNet', 'RegNetCfg']  # model_registry will add each entrypoint fn to this




import dataclasses
import logging
import os
from copy import deepcopy
from typing import Optional, Dict, Callable, Any, Tuple

from torch import nn as nn
from torch.hub import load_state_dict_from_url

from timm.models._features import FeatureListNet, FeatureHookNet
from timm.models._features_fx import FeatureGraphNet
from timm.models._helpers import load_state_dict
from timm.models._hub import has_hf_hub, download_cached_file, check_cached_file, load_state_dict_from_hf
from timm.models._manipulate import adapt_input_conv
from timm.models._pretrained import PretrainedCfg
from timm.models._prune import adapt_model_from_file
from timm.models._registry import get_pretrained_cfg

_logger = logging.getLogger(__name__)

# Global variables for rarely used pretrained checkpoint download progress and hash check.
# Use set_pretrained_download_progress / set_pretrained_check_hash functions to toggle.
_DOWNLOAD_PROGRESS = False
_CHECK_HASH = False
_USE_OLD_CACHE = int(os.environ.get('TIMM_USE_OLD_CACHE', 0)) > 0

__all__ = ['set_pretrained_download_progress', 'set_pretrained_check_hash', 'load_custom_pretrained', 'load_pretrained',
           'pretrained_cfg_for_features', 'resolve_pretrained_cfg', 'build_model_with_cfg']


def _resolve_pretrained_source(pretrained_cfg):
    cfg_source = pretrained_cfg.get('source', '')
    pretrained_url = pretrained_cfg.get('url', None)
    pretrained_file = pretrained_cfg.get('file', None)
    pretrained_sd = pretrained_cfg.get('state_dict', None)
    hf_hub_id = pretrained_cfg.get('hf_hub_id', None)

    # resolve where to load pretrained weights from
    load_from = ''
    pretrained_loc = ''
    if cfg_source == 'hf-hub' and has_hf_hub(necessary=True):
        # hf-hub specified as source via model identifier
        load_from = 'hf-hub'
        assert hf_hub_id
        pretrained_loc = hf_hub_id
    else:
        # default source == timm or unspecified
        if pretrained_sd:
            # direct state_dict pass through is the highest priority
            load_from = 'state_dict'
            pretrained_loc = pretrained_sd
            assert isinstance(pretrained_loc, dict)
        elif pretrained_file:
            # file load override is the second-highest priority if set
            load_from = 'file'
            pretrained_loc = pretrained_file
        else:
            old_cache_valid = False
            if _USE_OLD_CACHE:
                # prioritized old cached weights if exists and env var enabled
                old_cache_valid = check_cached_file(pretrained_url) if pretrained_url else False
            if not old_cache_valid and hf_hub_id and has_hf_hub(necessary=True):
                # hf-hub available as alternate weight source in default_cfg
                load_from = 'hf-hub'
                pretrained_loc = hf_hub_id
            elif pretrained_url:
                load_from = 'url'
                pretrained_loc = pretrained_url

    if load_from == 'hf-hub' and pretrained_cfg.get('hf_hub_filename', None):
        # if a filename override is set, return tuple for location w/ (hub_id, filename)
        pretrained_loc = pretrained_loc, pretrained_cfg['hf_hub_filename']
    return load_from, pretrained_loc


def set_pretrained_download_progress(enable=True):
    """ Set download progress for pretrained weights on/off (globally). """
    global _DOWNLOAD_PROGRESS
    _DOWNLOAD_PROGRESS = enable


def set_pretrained_check_hash(enable=True):
    """ Set hash checking for pretrained weights on/off (globally). """
    global _CHECK_HASH
    _CHECK_HASH = enable


def load_custom_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        load_fn: Optional[Callable] = None,
):
    r"""Loads a custom (read non .pth) weight file

    Downloads checkpoint file into cache-dir like torch.hub based loaders, but calls
    a passed in custom load fun, or the `load_pretrained` model member fn.

    If the object is already present in `model_dir`, it's deserialized and returned.
    The default value of `model_dir` is ``<hub_dir>/checkpoints`` where
    `hub_dir` is the directory returned by :func:`~torch.hub.get_dir`.

    Args:
        model: The instantiated model to load weights into
        pretrained_cfg (dict): Default pretrained model cfg
        load_fn: An external standalone fn that loads weights into provided model, otherwise a fn named
            'laod_pretrained' on the model will be called if it exists
    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        _logger.warning("Invalid pretrained config, cannot load weights.")
        return

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if not load_from:
        _logger.warning("No pretrained weights exist for this model. Using random initialization.")
        return
    if load_from == 'hf-hub':
        _logger.warning("Hugging Face hub not currently supported for custom load pretrained models.")
    elif load_from == 'url':
        pretrained_loc = download_cached_file(
            pretrained_loc,
            check_hash=_CHECK_HASH,
            progress=_DOWNLOAD_PROGRESS,
        )

    if load_fn is not None:
        load_fn(model, pretrained_loc)
    elif hasattr(model, 'load_pretrained'):
        model.load_pretrained(pretrained_loc)
    else:
        _logger.warning("Valid function to load pretrained weights is not available, using random initialization.")


def load_pretrained(
        model: nn.Module,
        pretrained_cfg: Optional[Dict] = None,
        num_classes: int = 1000,
        in_chans: int = 3,
        filter_fn: Optional[Callable] = None,
        strict: bool = True,
):
    """ Load pretrained checkpoint

    Args:
        model (nn.Module) : PyTorch model module
        pretrained_cfg (Optional[Dict]): configuration for pretrained weights / target dataset
        num_classes (int): num_classes for target model
        in_chans (int): in_chans for target model
        filter_fn (Optional[Callable]): state_dict filter fn for load (takes state_dict, model as args)
        strict (bool): strict load of checkpoint

    """
    pretrained_cfg = pretrained_cfg or getattr(model, 'pretrained_cfg', None)
    if not pretrained_cfg:
        raise RuntimeError("Invalid pretrained config, cannot load weights. Use `pretrained=False` for random init.")

    load_from, pretrained_loc = _resolve_pretrained_source(pretrained_cfg)
    if load_from == 'state_dict':
        _logger.info(f'Loading pretrained weights from state dict')
        state_dict = pretrained_loc  # pretrained_loc is the actual state dict for this override
    elif load_from == 'file':
        _logger.info(f'Loading pretrained weights from file ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict(pretrained_loc)
    elif load_from == 'url':
        _logger.info(f'Loading pretrained weights from url ({pretrained_loc})')
        if pretrained_cfg.get('custom_load', False):
            pretrained_loc = download_cached_file(
                pretrained_loc,
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
            )
            model.load_pretrained(pretrained_loc)
            return
        else:
            state_dict = load_state_dict_from_url(
                pretrained_loc,
                map_location='cpu',
                progress=_DOWNLOAD_PROGRESS,
                check_hash=_CHECK_HASH,
            )
    elif load_from == 'hf-hub':
        _logger.info(f'Loading pretrained weights from Hugging Face hub ({pretrained_loc})')
        if isinstance(pretrained_loc, (list, tuple)):
            state_dict = load_state_dict_from_hf(*pretrained_loc)
        else:
            state_dict = load_state_dict_from_hf(pretrained_loc)
    else:
        model_name = pretrained_cfg.get('architecture', 'this model')
        raise RuntimeError(f"No pretrained weights exist for {model_name}. Use `pretrained=False` for random init.")

    if filter_fn is not None:
        try:
            state_dict = filter_fn(state_dict, model)
        except TypeError as e:
            # for backwards compat with filter fn that take one arg
            state_dict = filter_fn(state_dict)

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None and in_chans != 3:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            try:
                state_dict[weight_name] = adapt_input_conv(in_chans, state_dict[weight_name])
                _logger.info(
                    f'Converted input conv {input_conv_name} pretrained weights from 3 to {in_chans} channel(s)')
            except NotImplementedError as e:
                del state_dict[weight_name]
                strict = False
                _logger.warning(
                    f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.')

    classifiers = pretrained_cfg.get('classifier', None)
    label_offset = pretrained_cfg.get('label_offset', 0)
    if classifiers is not None:
        if isinstance(classifiers, str):
            classifiers = (classifiers,)
        if num_classes != pretrained_cfg['num_classes']:
            for classifier_name in classifiers:
                # completely discard fully connected if model num_classes doesn't match pretrained weights
                state_dict.pop(classifier_name + '.weight', None)
                state_dict.pop(classifier_name + '.bias', None)
            strict = False
        elif label_offset > 0:
            for classifier_name in classifiers:
                # special case for pretrained weights with an extra background class in pretrained weights
                classifier_weight = state_dict[classifier_name + '.weight']
                state_dict[classifier_name + '.weight'] = classifier_weight[label_offset:]
                classifier_bias = state_dict[classifier_name + '.bias']
                state_dict[classifier_name + '.bias'] = classifier_bias[label_offset:]

    ret = model.load_state_dict(state_dict, strict=strict)
    print(ret)


def pretrained_cfg_for_features(pretrained_cfg):
    pretrained_cfg = deepcopy(pretrained_cfg)
    # remove default pretrained cfg fields that don't have much relevance for feature backbone
    to_remove = ('num_classes', 'classifier', 'global_pool')  # add default final pool size?
    for tr in to_remove:
        pretrained_cfg.pop(tr, None)
    return pretrained_cfg


def _filter_kwargs(kwargs, names):
    if not kwargs or not names:
        return
    for n in names:
        kwargs.pop(n, None)


def _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter):
    """ Update the default_cfg and kwargs before passing to model

    Args:
        pretrained_cfg: input pretrained cfg (updated in-place)
        kwargs: keyword args passed to model build fn (updated in-place)
        kwargs_filter: keyword arg keys that must be removed before model __init__
    """
    # Set model __init__ args that can be determined by default_cfg (if not already passed as kwargs)
    default_kwarg_names = ('num_classes', 'global_pool', 'in_chans')
    if pretrained_cfg.get('fixed_input_size', False):
        # if fixed_input_size exists and is True, model takes an img_size arg that fixes its input size
        default_kwarg_names += ('img_size',)

    for n in default_kwarg_names:
        # for legacy reasons, model __init__args uses img_size + in_chans as separate args while
        # pretrained_cfg has one input_size=(C, H ,W) entry
        if n == 'img_size':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[-2:])
        elif n == 'in_chans':
            input_size = pretrained_cfg.get('input_size', None)
            if input_size is not None:
                assert len(input_size) == 3
                kwargs.setdefault(n, input_size[0])
        else:
            default_val = pretrained_cfg.get(n, None)
            if default_val is not None:
                kwargs.setdefault(n, pretrained_cfg[n])

    # Filter keyword args for task specific model variants (some 'features only' models, etc.)
    _filter_kwargs(kwargs, names=kwargs_filter)


def resolve_pretrained_cfg(
        variant: str,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
) -> PretrainedCfg:
    model_with_tag = variant
    pretrained_tag = None
    if pretrained_cfg:
        if isinstance(pretrained_cfg, dict):
            # pretrained_cfg dict passed as arg, validate by converting to PretrainedCfg
            pretrained_cfg = PretrainedCfg(**pretrained_cfg)
        elif isinstance(pretrained_cfg, str):
            pretrained_tag = pretrained_cfg
            pretrained_cfg = None

    # fallback to looking up pretrained cfg in model registry by variant identifier
    if not pretrained_cfg:
        if pretrained_tag:
            model_with_tag = '.'.join([variant, pretrained_tag])
        pretrained_cfg = get_pretrained_cfg(model_with_tag)

    if not pretrained_cfg:
        _logger.warning(
            f"No pretrained configuration specified for {model_with_tag} model. Using a default."
            f" Please add a config to the model pretrained_cfg registry or pass explicitly.")
        pretrained_cfg = PretrainedCfg()  # instance with defaults

    pretrained_cfg_overlay = pretrained_cfg_overlay or {}
    if not pretrained_cfg.architecture:
        pretrained_cfg_overlay.setdefault('architecture', variant)
    pretrained_cfg = dataclasses.replace(pretrained_cfg, **pretrained_cfg_overlay)

    return pretrained_cfg


def build_model_with_cfg(
        model_cls: Callable,
        variant: str,
        pretrained: bool,
        pretrained_cfg: Optional[Dict] = None,
        pretrained_cfg_overlay: Optional[Dict] = None,
        model_cfg: Optional[Any] = None,
        feature_cfg: Optional[Dict] = None,
        pretrained_strict: bool = True,
        pretrained_filter_fn: Optional[Callable] = None,
        kwargs_filter: Optional[Tuple[str]] = None,
        **kwargs,
):
    """ Build model with specified default_cfg and optional model_cfg

    This helper fn aids in the construction of a model including:
      * handling default_cfg and associated pretrained weight loading
      * passing through optional model_cfg for models with config based arch spec
      * features_only model adaptation
      * pruning config / model adaptation

    Args:
        model_cls (nn.Module): model class
        variant (str): model variant name
        pretrained (bool): load pretrained weights
        pretrained_cfg (dict): model's pretrained weight/task config
        model_cfg (Optional[Dict]): model's architecture config
        feature_cfg (Optional[Dict]: feature extraction adapter config
        pretrained_strict (bool): load pretrained weights strictly
        pretrained_filter_fn (Optional[Callable]): filter callable for pretrained weights
        kwargs_filter (Optional[Tuple]): kwargs to filter before passing to model
        **kwargs: model args passed through to model __init__
    """
    pruned = kwargs.pop('pruned', False)
    features = False
    feature_cfg = feature_cfg or {}

    # resolve and update model pretrained config and model kwargs
    pretrained_cfg = resolve_pretrained_cfg(
        variant,
        pretrained_cfg=pretrained_cfg,
        pretrained_cfg_overlay=pretrained_cfg_overlay
    )

    # FIXME converting back to dict, PretrainedCfg use should be propagated further, but not into model
    pretrained_cfg = pretrained_cfg.to_dict()

    _update_default_kwargs(pretrained_cfg, kwargs, kwargs_filter)

    # Setup for feature extraction wrapper done at end of this fn
    if kwargs.pop('features_only', False):
        features = True
        feature_cfg.setdefault('out_indices', (0, 1, 2, 3, 4))
        if 'out_indices' in kwargs:
            feature_cfg['out_indices'] = kwargs.pop('out_indices')

    # Instantiate the model
    if model_cfg is None:
        model = model_cls(**kwargs)
    else:
        model = model_cls(cfg=model_cfg, **kwargs)
    model.pretrained_cfg = pretrained_cfg
    model.default_cfg = model.pretrained_cfg  # alias for backwards compat

    if pruned:
        model = adapt_model_from_file(model, variant)

    # For classification models, check class attr, then kwargs, then default to 1k, otherwise 0 for feats
    num_classes_pretrained = 0 if features else getattr(model, 'num_classes', kwargs.get('num_classes', 1000))
    if pretrained:
        load_pretrained(
            model,
            pretrained_cfg=pretrained_cfg,
            num_classes=num_classes_pretrained,
            in_chans=kwargs.get('in_chans', 3),
            filter_fn=pretrained_filter_fn,
            strict=pretrained_strict,
        )

    # Wrap the model in a feature extraction module if enabled
    if features:
        feature_cls = FeatureListNet
        output_fmt = getattr(model, 'output_fmt', None)
        if output_fmt is not None:
            feature_cfg.setdefault('output_fmt', output_fmt)
        if 'feature_cls' in feature_cfg:
            feature_cls = feature_cfg.pop('feature_cls')
            if isinstance(feature_cls, str):
                feature_cls = feature_cls.lower()
                if 'hook' in feature_cls:
                    feature_cls = FeatureHookNet
                elif feature_cls == 'fx':
                    feature_cls = FeatureGraphNet
                else:
                    assert False, f'Unknown feature class {feature_cls}'
        model = feature_cls(model, **feature_cfg)
        model.pretrained_cfg = pretrained_cfg_for_features(pretrained_cfg)  # add back pretrained cfg
        model.default_cfg = model.pretrained_cfg  # alias for rename backwards compat (default_cfg -> pretrained_cfg)

    return model




@dataclass
class RegNetCfg:
    depth: int = 21
    w0: int = 80
    wa: float = 42.63
    wm: float = 2.66
    group_size: int = 24
    bottle_ratio: float = 1.
    se_ratio: float = 0.
    group_min_ratio: float = 0.
    stem_width: int = 32
    downsample: Optional[str] = 'conv1x1'
    linear_out: bool = False
    preact: bool = False
    num_features: int = 0
    act_layer: Union[str, Callable] = 'relu'
    norm_layer: Union[str, Callable] = 'batchnorm'


def quantize_float(f, q):
    """Converts a float to the closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_widths_groups_comp(widths, bottle_ratios, groups, min_ratio=0.):
    """Adjusts the compatibility of widths and groups."""
    bottleneck_widths = [int(w * b) for w, b in zip(widths, bottle_ratios)]
    groups = [min(g, w_bot) for g, w_bot in zip(groups, bottleneck_widths)]
    if min_ratio:
        # torchvision uses a different rounding scheme for ensuring bottleneck widths divisible by group widths
        bottleneck_widths = [make_divisible(w_bot, g, min_ratio) for w_bot, g in zip(bottleneck_widths, groups)]
    else:
        bottleneck_widths = [quantize_float(w_bot, g) for w_bot, g in zip(bottleneck_widths, groups)]
    widths = [int(w_bot / b) for w_bot, b in zip(bottleneck_widths, bottle_ratios)]
    return widths, groups


def generate_regnet(width_slope, width_initial, width_mult, depth, group_size, quant=8):
    """Generates per block widths from RegNet parameters."""
    assert width_slope >= 0 and width_initial > 0 and width_mult > 1 and width_initial % quant == 0
    # TODO dWr scaling?
    # depth = int(depth * (scale ** 0.1))
    # width_scale = scale ** 0.4  # dWr scale, exp 0.8 / 2, applied to both group and layer widths
    widths_cont = np.arange(depth) * width_slope + width_initial
    width_exps = np.round(np.log(widths_cont / width_initial) / np.log(width_mult))
    widths = np.round(np.divide(width_initial * np.power(width_mult, width_exps), quant)) * quant
    num_stages, max_stage = len(np.unique(widths)), width_exps.max() + 1
    groups = np.array([group_size for _ in range(num_stages)])
    return widths.astype(int).tolist(), num_stages, groups.astype(int).tolist()


def downsample_conv(
        in_chs,
        out_chs,
        kernel_size=1,
        stride=1,
        dilation=1,
        norm_layer=None,
        preact=False,
):
    norm_layer = norm_layer or nn.BatchNorm2d
    kernel_size = 1 if stride == 1 and dilation == 1 else kernel_size
    dilation = dilation if kernel_size > 1 else 1
    if preact:
        return create_conv2d(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
        )
    else:
        return ConvNormAct(
            in_chs,
            out_chs,
            kernel_size,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
            apply_act=False,
        )


def downsample_avg(
        in_chs,
        out_chs,
        kernel_size=1,
        stride=1,
        dilation=1,
        norm_layer=None,
        preact=False,
):
    """ AvgPool Downsampling as in 'D' ResNet variants. This is not in RegNet space but I might experiment."""
    norm_layer = norm_layer or nn.BatchNorm2d
    avg_stride = stride if dilation == 1 else 1
    pool = nn.Identity()
    if stride > 1 or dilation > 1:
        avg_pool_fn = AvgPool2dSame if avg_stride == 1 and dilation > 1 else nn.AvgPool2d
        pool = avg_pool_fn(2, avg_stride, ceil_mode=True, count_include_pad=False)
    if preact:
        conv = create_conv2d(in_chs, out_chs, 1, stride=1)
    else:
        conv = ConvNormAct(in_chs, out_chs, 1, stride=1, norm_layer=norm_layer, apply_act=False)
    return nn.Sequential(*[pool, conv])


def create_shortcut(
        downsample_type,
        in_chs,
        out_chs,
        kernel_size,
        stride,
        dilation=(1, 1),
        norm_layer=None,
        preact=False,
):
    assert downsample_type in ('avg', 'conv1x1', '', None)
    if in_chs != out_chs or stride != 1 or dilation[0] != dilation[1]:
        dargs = dict(stride=stride, dilation=dilation[0], norm_layer=norm_layer, preact=preact)
        if not downsample_type:
            return None  # no shortcut, no downsample
        elif downsample_type == 'avg':
            return downsample_avg(in_chs, out_chs, **dargs)
        else:
            return downsample_conv(in_chs, out_chs, kernel_size=kernel_size, **dargs)
    else:
        return nn.Identity()  # identity shortcut (no downsample)


class Bottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            dilation=(1, 1),
            bottle_ratio=1,
            group_size=1,
            se_ratio=0.25,
            downsample='conv1x1',
            linear_out=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_block=None,
            drop_path_rate=0.,
    ):
        super(Bottleneck, self).__init__()
        act_layer = get_act_layer(act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        cargs = dict(act_layer=act_layer, norm_layer=norm_layer)
        self.conv1 = ConvNormAct(in_chs, bottleneck_chs, kernel_size=1, **cargs)
        self.conv2 = ConvNormAct(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
            drop_layer=drop_block,
            **cargs,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.conv3 = ConvNormAct(bottleneck_chs, out_chs, kernel_size=1, apply_act=False, **cargs)
        self.act3 = nn.Identity() if linear_out else act_layer()
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        nn.init.zeros_(self.conv3.bn.weight)

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        x = self.act3(x)
        return x


class PreBottleneck(nn.Module):
    """ RegNet Bottleneck

    This is almost exactly the same as a ResNet Bottlneck. The main difference is the SE block is moved from
    after conv3 to after conv2. Otherwise, it's just redefining the arguments for groups/bottleneck channels.
    """

    def __init__(
            self,
            in_chs,
            out_chs,
            stride=1,
            dilation=(1, 1),
            bottle_ratio=1,
            group_size=1,
            se_ratio=0.25,
            downsample='conv1x1',
            linear_out=False,
            act_layer=nn.ReLU,
            norm_layer=nn.BatchNorm2d,
            drop_block=None,
            drop_path_rate=0.,
    ):
        super(PreBottleneck, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        bottleneck_chs = int(round(out_chs * bottle_ratio))
        groups = bottleneck_chs // group_size

        self.norm1 = norm_act_layer(in_chs)
        self.conv1 = create_conv2d(in_chs, bottleneck_chs, kernel_size=1)
        self.norm2 = norm_act_layer(bottleneck_chs)
        self.conv2 = create_conv2d(
            bottleneck_chs,
            bottleneck_chs,
            kernel_size=3,
            stride=stride,
            dilation=dilation[0],
            groups=groups,
        )
        if se_ratio:
            se_channels = int(round(in_chs * se_ratio))
            self.se = SEModule(bottleneck_chs, rd_channels=se_channels, act_layer=act_layer)
        else:
            self.se = nn.Identity()
        self.norm3 = norm_act_layer(bottleneck_chs)
        self.conv3 = create_conv2d(bottleneck_chs, out_chs, kernel_size=1)
        self.downsample = create_shortcut(
            downsample,
            in_chs,
            out_chs,
            kernel_size=1,
            stride=stride,
            dilation=dilation,
            preact=True,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def zero_init_last(self):
        pass

    def forward(self, x):
        x = self.norm1(x)
        shortcut = x
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.conv2(x)
        x = self.se(x)
        x = self.norm3(x)
        x = self.conv3(x)
        if self.downsample is not None:
            # NOTE stuck with downsample as the attr name due to weight compatibility
            # now represents the shortcut, no shortcut if None, and non-downsample shortcut == nn.Identity()
            x = self.drop_path(x) + self.downsample(shortcut)
        return x


class RegStage(nn.Module):
    """Stage (sequence of blocks w/ the same output shape)."""

    def __init__(
            self,
            depth,
            in_chs,
            out_chs,
            stride,
            dilation,
            drop_path_rates=None,
            block_fn=Bottleneck,
            **block_kwargs,
    ):
        super(RegStage, self).__init__()
        self.grad_checkpointing = False

        first_dilation = 1 if dilation in (1, 2) else 2
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            block_in_chs = in_chs if i == 0 else out_chs
            block_dilation = (first_dilation, dilation)
            dpr = drop_path_rates[i] if drop_path_rates is not None else 0.
            name = "b{}".format(i + 1)
            self.add_module(
                name,
                block_fn(
                    block_in_chs,
                    out_chs,
                    stride=block_stride,
                    dilation=block_dilation,
                    drop_path_rate=dpr,
                    **block_kwargs,
                )
            )
            first_dilation = dilation

    def forward(self, x):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.children(), x)
        else:
            for block in self.children():
                x = block(x)
        return x


class RegNet(nn.Module):
    """RegNet-X, Y, and Z Models

    Paper: https://arxiv.org/abs/2003.13678
    Original Impl: https://github.com/facebookresearch/pycls/blob/master/pycls/models/regnet.py
    """

    def __init__(
            self,
            cfg: RegNetCfg,
            in_chans=3,
            num_classes=1000,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.,
            drop_path_rate=0.,
            zero_init_last=True,
            pruned_layers_stage3=0,
            pruned_layers_stage4=0,
            **kwargs,
    ):
        """

        Args:
            cfg (RegNetCfg): Model architecture configuration
            in_chans (int): Number of input channels (default: 3)
            num_classes (int): Number of classifier classes (default: 1000)
            output_stride (int): Output stride of network, one of (8, 16, 32) (default: 32)
            global_pool (str): Global pooling type (default: 'avg')
            drop_rate (float): Dropout rate (default: 0.)
            drop_path_rate (float): Stochastic depth drop-path rate (default: 0.)
            zero_init_last (bool): Zero-init last weight of residual path
            kwargs (dict): Extra kwargs overlayed onto cfg
        """
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        assert output_stride in (8, 16, 32)
        cfg = replace(cfg, **kwargs)  # update cfg with extra passed kwargs

        # Construct the stem
        stem_width = cfg.stem_width
        na_args = dict(act_layer=cfg.act_layer, norm_layer=cfg.norm_layer)
        if cfg.preact:
            self.stem = create_conv2d(in_chans, stem_width, 3, stride=2)
        else:
            self.stem = ConvNormAct(in_chans, stem_width, 3, stride=2, **na_args)
        self.feature_info = [dict(num_chs=stem_width, reduction=2, module='stem')]

        # Construct the stages
        prev_width = stem_width
        curr_stride = 2
        per_stage_args, common_args = self._get_stage_args(
            cfg,
            output_stride=output_stride,
            drop_path_rate=drop_path_rate,
        )
        assert len(per_stage_args) == 4
        block_fn = PreBottleneck if cfg.preact else Bottleneck
        for i, stage_args in enumerate(per_stage_args):
            stage_name = "s{}".format(i + 1)
            self.add_module(
                stage_name,
                RegStage(
                    in_chs=prev_width,
                    block_fn=block_fn,
                    **stage_args,
                    **common_args,
                )
            )
            prev_width = stage_args['out_chs']
            curr_stride *= stage_args['stride']
            self.feature_info += [dict(num_chs=prev_width, reduction=curr_stride, module=stage_name)]

        # Construct the head
        if cfg.num_features:
            self.final_conv = ConvNormAct(prev_width, cfg.num_features, kernel_size=1, **na_args)
            self.num_features = cfg.num_features
        else:
            final_act = cfg.linear_out or cfg.preact
            self.final_conv = get_act_layer(cfg.act_layer)() if final_act else nn.Identity()
            self.num_features = prev_width
        self.head = ClassifierHead(
            in_features=self.num_features,
            num_classes=num_classes,
            pool_type=global_pool,
            drop_rate=drop_rate,
        )

        if pruned_layers_stage3:
            # prune last few layers
            # print(self.s3)
            self.s3 = nn.Sequential(*list(self.s3.children())[:-(pruned_layers_stage3)])
            # print(self.s3)
        if pruned_layers_stage4 == 1:
            self.s4 = nn.Identity()
            # print(self.state_dict().keys())


        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    def _get_stage_args(self, cfg: RegNetCfg, default_stride=2, output_stride=32, drop_path_rate=0.):
        # Generate RegNet ws per block
        widths, num_stages, stage_gs = generate_regnet(cfg.wa, cfg.w0, cfg.wm, cfg.depth, cfg.group_size)

        # Convert to per stage format
        stage_widths, stage_depths = np.unique(widths, return_counts=True)
        stage_br = [cfg.bottle_ratio for _ in range(num_stages)]
        stage_strides = []
        stage_dilations = []
        net_stride = 2
        dilation = 1
        for _ in range(num_stages):
            if net_stride >= output_stride:
                dilation *= default_stride
                stride = 1
            else:
                stride = default_stride
                net_stride *= stride
            stage_strides.append(stride)
            stage_dilations.append(dilation)
        stage_dpr = np.split(np.linspace(0, drop_path_rate, sum(stage_depths)), np.cumsum(stage_depths[:-1]))

        # Adjust the compatibility of ws and gws
        stage_widths, stage_gs = adjust_widths_groups_comp(
            stage_widths, stage_br, stage_gs, min_ratio=cfg.group_min_ratio)
        arg_names = ['out_chs', 'stride', 'dilation', 'depth', 'bottle_ratio', 'group_size', 'drop_path_rates']
        per_stage_args = [
            dict(zip(arg_names, params)) for params in
            zip(stage_widths, stage_strides, stage_dilations, stage_depths, stage_br, stage_gs, stage_dpr)
        ]
        common_args = dict(
            downsample=cfg.downsample,
            se_ratio=cfg.se_ratio,
            linear_out=cfg.linear_out,
            act_layer=cfg.act_layer,
            norm_layer=cfg.norm_layer,
        )
        return per_stage_args, common_args

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',
            blocks=r'^s(\d+)' if coarse else r'^s(\d+)\.b(\d+)',
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in list(self.children())[1:-1]:
            s.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = self.final_conv(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name='', zero_init_last=False):
    if isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif zero_init_last and hasattr(module, 'zero_init_last'):
        module.zero_init_last()


import re

# Function to replace s3.b{int} with s3.{int-1}
def replace_string(input_string):
    def replace_match(match):
        # Extract the number after 'b'
        b_value = int(match.group(1))
        # Subtract 1 and return the new replacement string
        return f"s3.{b_value - 1}"

    # Regex pattern to match 's3.b{int}'
    pattern = r"s3\.b(\d+)"
    
    # Use re.sub to replace using the custom function
    return re.sub(pattern, replace_match, input_string)

# Example usage
# input_string = "s3.b1, s3.b2, s3.b10"
# result = replace_string(input_string)
# print(result)  # Output: s3.0, s3.1, s3.9


def _filter_fn(state_dict):
    state_dict = state_dict.get('model', state_dict)
    replaces = [
        ('f.a.0', 'conv1.conv'),
        ('f.a.1', 'conv1.bn'),
        ('f.b.0', 'conv2.conv'),
        ('f.b.1', 'conv2.bn'),
        ('f.final_bn', 'conv3.bn'),
        ('f.se.excitation.0', 'se.fc1'),
        ('f.se.excitation.2', 'se.fc2'),
        ('f.se', 'se'),
        ('f.c.0', 'conv3.conv'),
        ('f.c.1', 'conv3.bn'),
        ('f.c', 'conv3.conv'),
        ('proj.0', 'downsample.conv'),
        ('proj.1', 'downsample.bn'),
        ('proj', 'downsample.conv'),
    ]
    if 'classy_state_dict' in state_dict:
        # classy-vision & vissl (SEER) weights
        import re
        state_dict = state_dict['classy_state_dict']['base_model']['model']
        out = {}
        for k, v in state_dict['trunk'].items():
            k = k.replace('_feature_blocks.conv1.stem.0', 'stem.conv')
            k = k.replace('_feature_blocks.conv1.stem.1', 'stem.bn')
            k = re.sub(
                r'^_feature_blocks.res\d.block(\d)-(\d+)',
                lambda x: f's{int(x.group(1))}.b{int(x.group(2)) + 1}', k)
            k = re.sub(r's(\d)\.b(\d+)\.bn', r's\1.b\2.downsample.bn', k)
            for s, r in replaces:
                k = k.replace(s, r)
            out[k] = v
        for k, v in state_dict['heads'].items():
            if 'projection_head' in k or 'prototypes' in k:
                continue
            k = k.replace('0.clf.0', 'head.fc')
            out[k] = v
        return out
    if 'stem.0.weight' in state_dict:
        # torchvision weights
        import re
        out = {}
        for k, v in state_dict.items():
            k = k.replace('stem.0', 'stem.conv')
            k = k.replace('stem.1', 'stem.bn')
            k = re.sub(
                r'trunk_output.block(\d)\.block(\d+)\-(\d+)',
                lambda x: f's{int(x.group(1))}.b{int(x.group(3)) + 1}', k)
            for s, r in replaces:
                k = k.replace(s, r)
            k = k.replace('fc.', 'head.fc.')
            out[k] = v
        return out
    # print(state_dict.keys())

    new_state_dict = {}
    for k, v in state_dict.items():
        if 's3.b' in k:
            new_k = replace_string(k)
        else:
            new_k = k
        new_state_dict[new_k] = v

    return new_state_dict


# Model FLOPS = three trailing digits * 10^8
model_cfgs = dict(
    # RegNet-X
    regnetx_002=RegNetCfg(w0=24, wa=36.44, wm=2.49, group_size=8, depth=13),
    regnetx_004=RegNetCfg(w0=24, wa=24.48, wm=2.54, group_size=16, depth=22),
    regnetx_004_tv=RegNetCfg(w0=24, wa=24.48, wm=2.54, group_size=16, depth=22, group_min_ratio=0.9),
    regnetx_006=RegNetCfg(w0=48, wa=36.97, wm=2.24, group_size=24, depth=16),
    regnetx_008=RegNetCfg(w0=56, wa=35.73, wm=2.28, group_size=16, depth=16),
    regnetx_016=RegNetCfg(w0=80, wa=34.01, wm=2.25, group_size=24, depth=18),
    regnetx_032=RegNetCfg(w0=88, wa=26.31, wm=2.25, group_size=48, depth=25),
    regnetx_040=RegNetCfg(w0=96, wa=38.65, wm=2.43, group_size=40, depth=23),
    regnetx_064=RegNetCfg(w0=184, wa=60.83, wm=2.07, group_size=56, depth=17),
    regnetx_080=RegNetCfg(w0=80, wa=49.56, wm=2.88, group_size=120, depth=23),
    regnetx_120=RegNetCfg(w0=168, wa=73.36, wm=2.37, group_size=112, depth=19),
    regnetx_160=RegNetCfg(w0=216, wa=55.59, wm=2.1, group_size=128, depth=22),
    regnetx_320=RegNetCfg(w0=320, wa=69.86, wm=2.0, group_size=168, depth=23),

    # RegNet-Y
    regnety_002=RegNetCfg(w0=24, wa=36.44, wm=2.49, group_size=8, depth=13, se_ratio=0.25),
    regnety_004=RegNetCfg(w0=48, wa=27.89, wm=2.09, group_size=8, depth=16, se_ratio=0.25),
    regnety_006=RegNetCfg(w0=48, wa=32.54, wm=2.32, group_size=16, depth=15, se_ratio=0.25),
    regnety_008=RegNetCfg(w0=56, wa=38.84, wm=2.4, group_size=16, depth=14, se_ratio=0.25),
    regnety_008_tv=RegNetCfg(w0=56, wa=38.84, wm=2.4, group_size=16, depth=14, se_ratio=0.25, group_min_ratio=0.9),
    regnety_016=RegNetCfg(w0=48, wa=20.71, wm=2.65, group_size=24, depth=27, se_ratio=0.25),
    regnety_032=RegNetCfg(w0=80, wa=42.63, wm=2.66, group_size=24, depth=21, se_ratio=0.25),
    regnety_040=RegNetCfg(w0=96, wa=31.41, wm=2.24, group_size=64, depth=22, se_ratio=0.25),
    regnety_064=RegNetCfg(w0=112, wa=33.22, wm=2.27, group_size=72, depth=25, se_ratio=0.25),
    regnety_080=RegNetCfg(w0=192, wa=76.82, wm=2.19, group_size=56, depth=17, se_ratio=0.25),
    regnety_080_tv=RegNetCfg(w0=192, wa=76.82, wm=2.19, group_size=56, depth=17, se_ratio=0.25, group_min_ratio=0.9),
    regnety_120=RegNetCfg(w0=168, wa=73.36, wm=2.37, group_size=112, depth=19, se_ratio=0.25),
    regnety_160=RegNetCfg(w0=200, wa=106.23, wm=2.48, group_size=112, depth=18, se_ratio=0.25),
    regnety_320=RegNetCfg(w0=232, wa=115.89, wm=2.53, group_size=232, depth=20, se_ratio=0.25),
    regnety_640=RegNetCfg(w0=352, wa=147.48, wm=2.4, group_size=328, depth=20, se_ratio=0.25),
    regnety_1280=RegNetCfg(w0=456, wa=160.83, wm=2.52, group_size=264, depth=27, se_ratio=0.25),
    regnety_2560=RegNetCfg(w0=640, wa=230.83, wm=2.53, group_size=373, depth=27, se_ratio=0.25),
    #regnety_2560=RegNetCfg(w0=640, wa=124.47, wm=2.04, group_size=848, depth=27, se_ratio=0.25),

    # Experimental
    regnety_040_sgn=RegNetCfg(
        w0=96, wa=31.41, wm=2.24, group_size=64, depth=22, se_ratio=0.25,
        act_layer='silu', norm_layer=partial(GroupNormAct, group_size=16)),

    # regnetv = 'preact regnet y'
    regnetv_040=RegNetCfg(
        depth=22, w0=96, wa=31.41, wm=2.24, group_size=64, se_ratio=0.25, preact=True, act_layer='silu'),
    regnetv_064=RegNetCfg(
        depth=25, w0=112, wa=33.22, wm=2.27, group_size=72, se_ratio=0.25, preact=True, act_layer='silu',
        downsample='avg'),

    # RegNet-Z (unverified)
    regnetz_005=RegNetCfg(
        depth=21, w0=16, wa=10.7, wm=2.51, group_size=4, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=1024, act_layer='silu',
    ),
    regnetz_040=RegNetCfg(
        depth=28, w0=48, wa=14.5, wm=2.226, group_size=8, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=0, act_layer='silu',
    ),
    regnetz_040_h=RegNetCfg(
        depth=28, w0=48, wa=14.5, wm=2.226, group_size=8, bottle_ratio=4.0, se_ratio=0.25,
        downsample=None, linear_out=True, num_features=1536, act_layer='silu',
    ),
)


def _create_regnet(variant, pretrained, **kwargs):
    return build_model_with_cfg(
        RegNet, variant, pretrained,
        model_cfg=model_cfgs[variant],
        pretrained_filter_fn=_filter_fn,
        pretrained_strict=True,
        **kwargs)


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'test_input_size': (3, 288, 288), 'crop_pct': 0.95, 'test_crop_pct': 1.0,
        'interpolation': 'bicubic', 'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'stem.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    # timm trained models
    'regnety_064_pruned_3_1.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_064_ra3-aa26dc7d.pth'),
    'regnety_080_pruned_30p.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_080_ra3-1fdc4344.pth'),
    'regnety_080_pruned_2_1.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_080_ra3-1fdc4344.pth'),
    'regnety_080_pruned_3_1.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_080_ra3-1fdc4344.pth'),
    'regnety_080_pruned_3_0.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_080_ra3-1fdc4344.pth'),
    'regnety_080_pruned_4_0.ra3_in1k': _cfg(
        hf_hub_id='timm/',
        url='https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-tpu-weights/regnety_080_ra3-1fdc4344.pth'),
})


@register_model
def regnety_064_pruned_3_1(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-6.4GF"""
    return _create_regnet('regnety_064', pretrained, pruned_layers_stage3=3, pruned_layers_stage4=1, **kwargs)

@register_model
def regnety_080_pruned_30p(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, pruned_percent=30, **kwargs)


@register_model
def regnety_080_pruned_2_1(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, pruned_layers_stage3=2, pruned_layers_stage4=1, **kwargs)

@register_model
def regnety_080_pruned_3_1(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, pruned_layers_stage3=3, pruned_layers_stage4=1, **kwargs)

@register_model
def regnety_080_pruned_3_0(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, pruned_layers_stage3=3, pruned_layers_stage4=0, **kwargs)

@register_model
def regnety_080_pruned_4_0(pretrained=False, **kwargs) -> RegNet:
    """RegNetY-8.0GF"""
    return _create_regnet('regnety_080', pretrained, pruned_layers_stage3=4, pruned_layers_stage4=0, **kwargs)

