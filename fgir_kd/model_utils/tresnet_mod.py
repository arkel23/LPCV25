"""
TResNet: High Performance GPU-Dedicated Architecture
https://arxiv.org/pdf/2003.13630.pdf

Original model: https://github.com/mrT23/TResNet

"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from timm.layers import SpaceToDepth, BlurPool2d, ClassifierHead, SEModule,\
    ConvNormActAa, ConvNormAct, DropPath
from timm.models._builder import build_model_with_cfg
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import register_model, generate_default_cfgs, register_model_deprecations

__all__ = ['TResNet']  # model_registry will add each entrypoint fn to this


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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            use_se=True,
            aa_layer=None,
            drop_path_rate=0.
    ):
        super(BasicBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        act_layer = partial(nn.LeakyReLU, negative_slope=1e-3)

        if stride == 1:
            self.conv1 = ConvNormAct(inplanes, planes, kernel_size=3, stride=1, act_layer=act_layer)
        else:
            self.conv1 = ConvNormActAa(
                inplanes, planes, kernel_size=3, stride=2, act_layer=act_layer, aa_layer=aa_layer)

        self.conv2 = ConvNormAct(planes, planes, kernel_size=3, stride=1, apply_act=False, act_layer=None)
        self.act = nn.ReLU(inplace=True)

        rd_chs = max(planes * self.expansion // 4, 64)
        self.se = SEModule(planes * self.expansion, rd_channels=rd_chs) if use_se else None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.drop_path(out) + shortcut
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
            self,
            inplanes,
            planes,
            stride=1,
            downsample=None,
            use_se=True,
            act_layer=None,
            aa_layer=None,
            drop_path_rate=0.,
    ):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.stride = stride
        act_layer = act_layer or partial(nn.LeakyReLU, negative_slope=1e-3)

        self.conv1 = ConvNormAct(
            inplanes, planes, kernel_size=1, stride=1, act_layer=act_layer)
        if stride == 1:
            self.conv2 = ConvNormAct(
                planes, planes, kernel_size=3, stride=1, act_layer=act_layer)
        else:
            self.conv2 = ConvNormActAa(
                planes, planes, kernel_size=3, stride=2, act_layer=act_layer, aa_layer=aa_layer)

        reduction_chs = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, rd_channels=reduction_chs) if use_se else None

        self.conv3 = ConvNormAct(
            planes, planes * self.expansion, kernel_size=1, stride=1, apply_act=False, act_layer=None)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            shortcut = self.downsample(x)
        else:
            shortcut = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None:
            out = self.se(out)
        out = self.conv3(out)
        out = self.drop_path(out) + shortcut
        out = self.act(out)
        return out


class TResNet(nn.Module):
    def __init__(
            self,
            layers,
            in_chans=3,
            num_classes=1000,
            width_factor=1.0,
            v2=False,
            global_pool='fast',
            drop_rate=0.,
            drop_path_rate=0.,
    ):
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        super(TResNet, self).__init__()

        aa_layer = BlurPool2d
        act_layer = nn.LeakyReLU

        # TResnet stages
        self.inplanes = int(64 * width_factor)
        self.planes = int(64 * width_factor)
        if v2:
            self.inplanes = self.inplanes // 8 * 8
            self.planes = self.planes // 8 * 8

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        conv1 = ConvNormAct(in_chans * 16, self.planes, stride=1, kernel_size=3, act_layer=act_layer)

        layer1 = self._make_layer(
            Bottleneck if v2 else BasicBlock,
            self.planes, layers[0], stride=1, use_se=True, aa_layer=aa_layer, drop_path_rate=dpr[0])
        layer2 = self._make_layer(
            Bottleneck if v2 else BasicBlock,
            self.planes * 2, layers[1], stride=2, use_se=True, aa_layer=aa_layer, drop_path_rate=dpr[1])
        layer3 = self._make_layer(
            Bottleneck,
            self.planes * 4, layers[2], stride=2, use_se=True, aa_layer=aa_layer, drop_path_rate=dpr[2])
        layer4 = self._make_layer(
            Bottleneck,
            self.planes * 8, layers[3], stride=2, use_se=False, aa_layer=aa_layer, drop_path_rate=dpr[3])

        # body
        self.body = nn.Sequential(OrderedDict([
            ('s2d', SpaceToDepth()),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4),
        ]))

        # self.body = nn.Sequential(OrderedDict([
        #     ('s2d', SpaceToDepth()),
        #     ('conv1', conv1),
        #     ('layer1', layer1),
        #     ('layer2', layer2),
        #     ('layer3', layer3),
        #     ('layer4', layer4),
        # ]))

        self.feature_info = [
            dict(num_chs=self.planes, reduction=2, module=''),  # Not with S2D?
            dict(num_chs=self.planes * (Bottleneck.expansion if v2 else 1), reduction=4, module='body.layer1'),
            dict(num_chs=self.planes * 2 * (Bottleneck.expansion if v2 else 1), reduction=8, module='body.layer2'),
            dict(num_chs=self.planes * 4 * Bottleneck.expansion, reduction=16, module='body.layer3'),
            dict(num_chs=self.planes * 8 * Bottleneck.expansion, reduction=32, module='body.layer4'),
        ]

        # head
        self.num_features = (self.planes * 8) * Bottleneck.expansion
        self.head = ClassifierHead(self.num_features, num_classes, pool_type=global_pool, drop_rate=drop_rate)

        # model initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

        # residual connections special initialization
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.conv2.bn.weight)
            if isinstance(m, Bottleneck):
                nn.init.zeros_(m.conv3.bn.weight)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True, aa_layer=None, drop_path_rate=0.):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True, count_include_pad=False))
            layers += [ConvNormAct(
                self.inplanes, planes * block.expansion, kernel_size=1, stride=1, apply_act=False, act_layer=None)]
            downsample = nn.Sequential(*layers)

        layers = []
        for i in range(blocks):
            layers.append(block(
                self.inplanes,
                planes,
                stride=stride if i == 0 else 1,
                downsample=downsample if i == 0 else None,
                use_se=use_se,
                aa_layer=aa_layer,
                drop_path_rate=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate,
            ))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(stem=r'^body\.conv1', blocks=r'^body\.layer(\d+)' if coarse else r'^body\.layer(\d+)\.(\d+)')
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = self.body.s2d(x)
        #     x = self.body.conv1(x)
        #     x = checkpoint_seq([
        #         self.body.layer1,
        #         self.body.layer2,
        #         self.body.layer3,
        #         self.body.layer4],
        #         x, flatten=True)
        # else:
            # x = self.body(x)
        x = self.body.s2d(x)
        x = self.body.conv1(x)
        x1 = self.body.layer1(x)
        x2 = self.body.layer2(x1)
        x3 = self.body.layer3(x2)
        x4 = self.body.layer4(x3)
        return x3, x4

    def forward_head(self, x, pre_logits: bool = False):
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    if 'body.conv1.conv.weight' in state_dict:
        return state_dict

    import re
    state_dict = state_dict.get('model', state_dict)
    state_dict = state_dict.get('state_dict', state_dict)
    out_dict = {}
    for k, v in state_dict.items():
        k = re.sub(r'conv(\d+)\.0.0', lambda x: f'conv{int(x.group(1))}.conv', k)
        k = re.sub(r'conv(\d+)\.0.1', lambda x: f'conv{int(x.group(1))}.bn', k)
        k = re.sub(r'conv(\d+)\.0', lambda x: f'conv{int(x.group(1))}.conv', k)
        k = re.sub(r'conv(\d+)\.1', lambda x: f'conv{int(x.group(1))}.bn', k)
        k = re.sub(r'downsample\.(\d+)\.0', lambda x: f'downsample.{int(x.group(1))}.conv', k)
        k = re.sub(r'downsample\.(\d+)\.1', lambda x: f'downsample.{int(x.group(1))}.bn', k)
        if k.endswith('bn.weight'):
            # convert weight from inplace_abn to batchnorm
            v = v.abs().add(1e-5)
        out_dict[k] = v
    return out_dict


def _create_tresnet(variant, pretrained=False, **kwargs):
    return build_model_with_cfg(
        TResNet,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(out_indices=(1, 2, 3, 4), flatten_sequential=True),
        **kwargs,
    )


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': (0., 0., 0.), 'std': (1., 1., 1.),
        'first_conv': 'body.conv1.conv', 'classifier': 'head.fc',
        **kwargs
    }


default_cfgs = generate_default_cfgs({
    'tresnet_m_mod.miil_in21k_ft_in1k': _cfg(
        hf_hub_id='timm/tresnet_m.miil_in21k_ft_in1k',
    ),
    'tresnet_m_mod.miil_in21k': _cfg(
        hf_hub_id='timm/ttresnet_m_mod.miil_in21k',
        num_classes=11221,
    ),
})


@register_model
def tresnet_m_mod(pretrained=False, **kwargs) -> TResNet:
    model_args = dict(layers=[3, 4, 11, 3])
    return _create_tresnet('tresnet_m_mod', pretrained=pretrained, **dict(model_args, **kwargs))

