import re
from copy import deepcopy
from types import SimpleNamespace

import timm
import torch
import torch.nn as nn

from .modules_others import van_dict, ViT, ViTConfig, Head, CAL, CRD
from .regnety_mod import *


VITS = [
    'vit_n16', 'vit_m16', 'vit_t4', 'vit_t8', 'vit_t16', 'vit_t32',
    'vit_s8', 'vit_s16', 'vit_s32',
    'vit_b8', 'vit_b16', 'vit_b32', 'vit_l16', 'vit_l32', 'vit_h14']


def build_model(args, teacher=False, student=False):
    if teacher:
        from fgir_kd.model_utils.modules_others.swin import swin_base_patch4_window7_224, swin_large_patch4_window7_224
        args = deepcopy(args)
        args.model_name = args.model_name_teacher
        args.ckpt_path = args.ckpt_path_teacher
    elif student:
        args = deepcopy(args)
        args.image_size = args.student_image_size if args.student_image_size else args.image_size

    # initiates model and loss
    if (args.model_name in VITS or 'van' in args.model_name or
        args.model_name in timm.list_models() or args.model_name in timm.list_models(pretrained=True)):
        model = ClassifierModel(args, teacher, student)
    else:
        raise NotImplementedError

    if args.ckpt_path:
        load_model_compatibility_mode(args, model)

    if teacher and not args.train_both:
        freeze_backbone(model)

    if args.distributed:
        model.cuda()
    else:
        model.to(args.device)

    print(f'Initialized classifier: {args.model_name}')
    return model


def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    print('Total parameters (M): ', sum([p.numel() for p in model.parameters()]) / (1e6))
    print('Trainable parameters (M): ', sum([p.numel() for p in model.parameters() if p.requires_grad]) / (1e6))
    return 0


def convert_cal_student(state_dict):
    new_state_dict = {}

    for k, v in state_dict.items():

        if 'dfsm.' in k:
            # expected_missing_keys += ['model.dfsm.1.weight', 'model.dfsm.1.bias']
            continue

        new_k = k.replace('model.encoder.0.', 'model.')

        new_state_dict[new_k] = v

    return new_state_dict


def load_model_compatibility_mode(args, model):
    state_dict = torch.load(
        args.ckpt_path, map_location=torch.device('cpu'))['model']
    expected_missing_keys = []

    # retrocompatibility with prev experiments
    if 'model.head.head.weight' in state_dict.keys():
        state_dict['head.head.weight'] = state_dict.pop('model.head.head.weight')
        state_dict['head.head.bias'] = state_dict.pop('model.head.head.bias')
    elif 'head.weight' in state_dict.keys():
        state_dict['head.head.weight'] = state_dict.pop('head.weight')
        state_dict['head.head.bias'] = state_dict.pop('head.bias')

    # saved when using distributed training has an additional module. at the start of the keys
    if list(state_dict.keys())[0].startswith('module.'):
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                new_k = k.replace('module.', '', 1)
                state_dict[new_k] = state_dict.pop(k)

    if args.transfer_learning_cal:
        state_dict = convert_cal_student(state_dict)

    if args.transfer_learning:
        # modifications to load partial state dict
        if ('model.head.weight' in state_dict):
            expected_missing_keys += ['model.head.weight', 'model.head.bias']
        for key in expected_missing_keys:
            state_dict.pop(key)
    ret = model.load_state_dict(state_dict, strict=False)
    print('''Missing keys when loading pretrained weights: {}
            Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
    print('Unexpected keys when loading pretrained weights: {}'.format(
        ret.unexpected_keys))
    print('Loaded from custom checkpoint.')
    return 0


def get_backbone(args):
    args.classifier = 'pool'  # def, use cls for vit/deit if not cal

    if args.model_name in VITS:
        if not (args.selector == 'cal' or args.transfer_learning_cal):
            args.classifier = 'cls'
        # args.classifier = 'pool' if args.selector == 'cal' else 'cls'
        # init default config
        cfg = ViTConfig(model_name=args.model_name, image_size=args.image_size)
        cfg.classifier = args.classifier
        cfg.calc_dims()

        # init model
        model = ViT(cfg, pretrained=args.pretrained)

    elif 'deit' in args.model_name:
        # args.classifier = 'pool' if args.selector == 'cal' else 'cls'
        if not (args.selector == 'cal' or args.transfer_learning_cal):
            args.classifier = 'cls'
        cls = True if args.classifier == 'cls' else False
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0, class_token=cls,
            img_size=args.image_size, drop_path_rate=args.sd, global_pool='')
    elif 'van' in args.model_name:
        model = van_dict[args.model_name](
            pretrained=args.pretrained,img_size=args.image_size, drop_path_rate=args.sd)
    elif 'vgg' in args.model_name:
        model = timm.create_model(args.model_name, pretrained=args.pretrained,
                                  num_classes=0, global_pool='',
                                  pre_logits=False if args.selector == 'cal' else True)
    elif 'lrresnet' in args.model_name:
        drop_cls_token = True if args.selector == 'cal' else False
        img_size = args.student_image_size if args.student_image_size else args.image_size
        model = timm.create_model(
            args.model_name, pretrained=False, num_classes=0, drop_path_rate=args.sd,
            global_pool='', mlp_ratio=args.mlp_ratio, img_size=img_size,
            pos_embedding_type=args.pos_embedding_type, drop_cls_token=drop_cls_token)
    elif any(model in args.model_name for model in [
        'lcnet', 'resnet', 'convnext', 'densenet', 'resn',
        'efficientnet', 'ception', 'rexn', 'regn', 'focalnet',
        'hrnet', 'hgnet', 'rdnet']):
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0,
            drop_path_rate=args.sd, global_pool='')
    else:
        model = timm.create_model(
            args.model_name, pretrained=args.pretrained, num_classes=0,
            img_size=args.image_size, drop_path_rate=args.sd, global_pool='')

    return model


def get_layers(args, model):
    if args.layer_names:
        args.layer_names = args.layer_names[-args.num_layers:]
        return args.layers_names

    if 'convnext' in args.model_name_teacher or 'resnetv2' in args.model_name_teacher:
        pattern = re.compile(r'.*stages\.\d+\.blocks\.\d+$')

    elif 'regnet' in args.model_name_teacher:
        pattern = re.compile(r'.*s\d+\.b\d+$')

    elif 'resnet' in args.model_name_teacher:
        pattern = re.compile(r'.*layer\d+\.\d+$')

    elif 'swin' in args.model_name_teacher:
        pattern = re.compile(r'.*layers\.\d+\.blocks\.\d+$')

    elif 'beitv2' in args.model_name_teacher or 'deit' in args.model_name_teacher:
        pattern = re.compile(r'.*blocks\.\d+$')

    elif 'vit' in args.model_name_teacher:
        pattern = re.compile(r'.*encoder.blocks\.\d+$')

    elif 'van' in args.model_name_teacher:
        pattern = re.compile(r'.*block\d+\.\d+')

    elif 'vgg' in args.model_name_teacher:
        pattern = re.compile(r'.*features\.\d+')

    else:
        raise NotImplementedError
    
    all_names = []
    for name, _ in model.named_modules():
        all_names.append(name)

    layers = [l for l in all_names if pattern.match(l)]

    if args.selector == 'cal' and any([kw in args.model_name for kw in ('vit', 'deit', 'beit', 'swin', 'van')]):
        layers = ['encoder.0.' + l for l in layers]
        layers.append('encoder.1')
    elif args.selector == 'cal':
        layers = ['encoder.' + l for l in layers]
        layers.append('encoder')

    print('Stage output layers in model: ', layers)

    layers = layers[-args.num_layers:]
    args.layer_names = layers

    print('Layers to use: ', layers)

    return layers


class ClassifierModel(nn.Module):
    def __init__(self, args, teacher=False, student=False):
        super(ClassifierModel, self).__init__()

        model = get_backbone(args)
        img_size = args.student_image_size if (args.student_image_size and student) else args.image_size
        s, d, bsd = self.get_out_features(img_size, model)

        if teacher:
            layers = get_layers(args, model)
        else:
            layers = None

        if args.selector == 'cal':
            self.model = CAL(
                model=model,
                seq_len=s,
                output_size=d,
                num_classes=args.num_classes,
                bsd=bsd,
                device=args.device,
                teacher=teacher,
                student=student,
                tgda=args.tgda,
                kd_aux_loss=args.kd_aux_loss,
                num_images=args.num_images_train,
                cont_negatives=args.cont_negatives,
                cont_temp=args.cont_temp,
                cont_loss=args.cont_loss,
                image_size=args.image_size,
                layers=layers,
                pooling_function=args.pooling_function,
                pool_size=args.pool_size,
                disc_feats_norm=args.disc_feats_norm,
                disc_feats_sign_sqrt=args.disc_feats_sign_sqrt,
                if_channels=args.if_channels,
                mlp_ratio=args.mlp_ratio,
                pre_resize_factor=args.pre_resize_factor,
                student_image_size=args.student_image_size,
                cal_ap_only=args.cal_ap_only,
            )

        else:
            self.model = model
            self.head = Head(args.classifier, d, args.num_classes, bsd)

            if teacher and args.kd_aux_loss == 'crd':
                self.return_inter_feats = True
                self.if_channels = d
            elif student and args.kd_aux_loss == 'crd':
                self.loss_weight = args.loss_kd_aux_weight
                self.crd = CRD(
                    args.num_images_train, d, args.if_channels, int(d * args.mlp_ratio),
                    'gap', bsd, args.cont_negatives, args.cont_temp
                )

        self.cfg = SimpleNamespace(**{'seq_len': s, 'hidden_size': d})

    @torch.no_grad()
    def get_out_features(self, image_size, model):
        x = torch.rand(2, 3, image_size, image_size)
        x = model(x)

        if len(x.shape) == 3:
            b, s, d = x.shape
            bsd = True
        elif len(x.shape) == 4:
            b, d, h, w = x.shape

            if h != w:
                raise NotImplementedError
                # b, h, w, c is the actual shape
                temp1 = h
                temp2 = w
                temp3 = d
                h = temp1
                w = temp2
                d = temp3

            s = h * w
            bsd = False

        print('Output feature shape: ', x.shape)

        return s, d, bsd

    def forward(self, images, targets=None, output_t=None, idx=None, sample_idx=None):
        if hasattr(self, 'head'):
            features = self.model(images)
            out = self.head(features)

            if hasattr(self, 'return_inter_feats') and self.training:
                return out, features
            elif hasattr(self, 'crd') and self.training:
                loss = self.crd(features, output_t[1], idx, sample_idx)
                return out, loss

            return out

        else:
            out = self.model(images, targets, output_t, idx, sample_idx)

        return out
