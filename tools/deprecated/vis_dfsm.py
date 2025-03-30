import os
from typing import List
from functools import partial
from contextlib import suppress

import cv2
import wandb
import matplotlib.pyplot as plt
from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
from einops import reduce
from torchcam import methods

from fgir_kd.data_utils.build_dataloaders import build_dataloaders
from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.model_utils.build_model import build_model
from fgir_kd.train_utils.misc_utils import set_random_seed
from fgir_kd.train_utils.save_vis_images import inverse_normalize


MODELS_DIC = {
    'vit_b16': 'ViT',
}

DATASETS_DIC = {
    'aircraft': 'Aircraft',
    'cars': 'Cars',
    'cotton': 'Cotton',
    'cub': 'CUB',
    'dafb': 'DAFB',
    'dogs': 'Dogs',
    'flowers': 'Flowers',
    'food': 'Food',
    'inat17': 'iNat17',
    'moe': 'Moe',
    'nabirds': 'NABirds',
    'pets': 'Pets',
    'soyageing': 'SoyAgeing',
    'soyageingr1': 'SoyAgeingR1',
    'soyageingr3': 'SoyAgeingR3',
    'soyageingr4': 'SoyAgeingR4',
    'soyageingr5': 'SoyAgeingR5',
    'soyageingr6': 'SoyAgeingR6',
    'soygene': 'SoyGene',
    'soyglobal': 'SoyGlobal',
    'soylocal': 'SoyLocal',
    'vegfru': 'VegFru',
}

SETTINGS_DIC = {
    'scratch': 'Scratch',
    'pt': 'Pre-trained',
    'ft': 'Fine-tuned',
}


def adjust_args_general(args):
    selector = f'_{args.selector}' if args.selector else ''

    args.run_name = '{}_{}{}_{}_{}'.format(
        args.dataset_name, args.model_name, selector, args.setting, args.serial
    )

    args.results_dir = os.path.join(args.results_inference, args.run_name)
    os.makedirs(args.results_dir, exist_ok=True)
    return args


class VisHook:
    def __init__(self,
                 model: nn.Module,
                 model_layers: List[str] = None,
                 vis_mask: str = None,
                 device: str ='cpu'):
        """
        :param model: (nn.Module) Neural Network 1
        :param model_layers: (List) List of layers to extract features from
        :param device: Device to run the model
        """

        self.model = model

        self.device = device

        self.model_info = {}

        self.model_info['Layers'] = []

        self.model_features = {}

        self.model_layers = model_layers

        self.vis_mask = vis_mask

        if vis_mask and 'CAM' in vis_mask:
            # for _, param in self.model.named_parameters():
            #    param.requires_grad = True

            self.model = self.model.to(self.device)

            for name, layer in self.model.named_modules():
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]

            self.extractor = methods.__dict__[vis_mask](self.model, target_layer=self.model_info['Layers'][-1], enable_hooks=True)

        else:
            self._insert_hooks()
            self.model = self.model.to(self.device)

            self.model.eval()

        print(self.model_info)


    def _log_layer(self,
                   model: str,
                   name: str,
                   layer: nn.Module,
                   inp: torch.Tensor,
                   out: torch.Tensor):

        if model == "model":
            self.model_features[name] = out
        else:
            raise RuntimeError("Unknown model name for _log_layer.")


    def _insert_hooks(self):
        # Model 1
        for name, layer in self.model.named_modules():
            if self.model_layers is not None:
                if name in self.model_layers:
                    self.model_info['Layers'] += [name]
                    layer.register_forward_hook(partial(self._log_layer, "model", name))
            else:
                self.model_info['Layers'] += [name]
                layer.register_forward_hook(partial(self._log_layer, "model", name))


    def extract_features(self, images) -> None:
        """
        Computes the attention rollout for the image(s)
        :param x: (input tensor)
        """
        self.model_features = {}

        images = images.to(self.device)

        if self.vis_mask and 'CAM' in self.vis_mask:
            masks = []
            for i, image in enumerate(images):
                self.extractor._hooks_enabled = True
                self.model.zero_grad()

                preds = self.model(image.unsqueeze(0))

                _, class_idx = torch.max(preds, dim=-1)
                activation_map = self.extractor(class_idx.item(), preds[i:i+1])[0]

                self.extractor.remove_hooks()
                self.extractor._hooks_enabled = False

                masks.append(activation_map)

            return masks

        preds = self.model(images)

        return self.model_features


    def save_vis(self, args, loader, split):
        images, _ = next(iter(loader))

        features = self.extract_features(images)
        regs = 4 if 'reg4' in args.model_name else 0
        masks = calc_masks(features, self.vis_mask, regs=regs)

        masked_imgs = []
        for i in range(images.shape[0]):
            img_unnorm = inverse_normalize(images[i].detach().clone(), args.custom_mean_std)
            img_masked = apply_mask(img_unnorm, masks[i], args.vis_mask_pow)
            masked_imgs.append(img_masked)

        number_imgs = len(masked_imgs)
        ncols = args.vis_cols if args.vis_cols else int(number_imgs ** 0.5)
        nrows = number_imgs // ncols
        number_imgs = ncols * nrows

        fig = plt.figure(figsize=(ncols, nrows))
        grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols),
                        axes_pad=(0.01, 0.01), direction='row', aspect=True)

        for i, (ax, np_arr) in enumerate(zip(grid, masked_imgs)):
            ax.axis('off')
            ax.imshow(np_arr)

        save_images(fig, self.vis_mask, args.vis_mask_pow, split, args.results_dir, args.debugging)
            
        return 0


def save_images(fig, vis_mask, power, split, output_dir, debugging=True):
    pow = '_power' if power else ''
    fn = f'{vis_mask}{pow}_{split}.png'
    fp = os.path.join(output_dir, fn)
    fig.savefig(fp, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print('Saved ', fp)

    if not debugging:
        wandb.log({f'{split}': wandb.Image(fig)})

    return 0


def calc_masks(features, vis_mask='rollout', def_rollout_end=4, regs=0):
    if vis_mask is None:
        bs = list(features.values())[0].shape[0]
        masks = [None for _ in range(bs)]
        # return 


    elif 'rollout' in vis_mask or 'attention' in vis_mask:
        features = {k: v for k, v in features.items() if 'attn' in k}
        print(features.keys())
        features = list(features.values())


        if 'rollout' in vis_mask:
            splits = vis_mask.split('_')
            if len(splits) == 1:
                rollout_start = 0
                rollout_end = int(def_rollout_end)
            elif len(splits) == 2:
                rollout_start = 0
                rollout_end = int(splits[-1])
            elif len(splits) == 3:
                rollout_start = int(splits[1])
                rollout_end = int(splits[-1])
            else:
                raise NotImplementedError

            # input: list of length L with each element shape: B, NH, S, S
            # use only first 4 (should be similar enough to full rollout)
            # because scores after 4 should have different shape
            # output: B, S, S
            # select 1st token attention for the rest []:, 0, 1:] -> B, S-1

            attention = features[rollout_start:rollout_end]
            attention = attention_rollout(attention)
            if regs > 0:
                masks = attention[:, 0, 1+regs:]
            else:
                masks = attention[:, 0, 1:]


        elif 'attention' in vis_mask:
            splits = vis_mask.split('_')
            if len(splits) == 1:
                layer = 0
            elif len(splits) == 2:
                layer = int(splits[-1])
            else:
                raise NotImplementedError

            attention = features[layer]
            attention = reduce(attention, 'b h s1 s2 -> b s1 s2', 'mean')
            if regs > 0:
                masks = attention[:, 0, 1+regs:]
            else:
                masks = attention[:, 0, 1:]


    elif vis_mask == 'gls':
        features = {k: v for k, v in features.items() if 'norm' in k}
        print(features.keys())
        features = list(features.values())[-1]

        fh = int(features.shape[1] ** 0.5)
        if fh ** 2 == features.shape[1]:
            g = reduce(features, 'b s d -> b d', 'mean')
            l = features
        else:
            g = features[:, :1]
            l = features[:, 1:]

        masks = F.cosine_similarity(g, l, dim=-1)


    elif 'bap' in vis_mask:
        print(features.keys())
        features = list(features.values())[0]

        if vis_mask == 'bap_avg':
            masks = reduce(features, 'b ac ah aw -> b (ah aw)', 'mean')
        else:
            map = int(vis_mask.split('_')[-1])
            masks = reduce(features[:, map], 'b ah aw -> b (ah aw)', 'mean')


    elif 'CAM' in vis_mask:
        masks = torch.cat(features, dim=0)
        masks = rearrange(masks, 'b ah aw -> b (ah aw)')


    else:
        raise NotImplementedError


    return masks


def apply_mask(img, mask, power=False, color=True):
    '''
    img are pytorch tensors of size C x S1 x S2, range 0-255 (unnormalized)
    mask are pytorch tensors of size (S1 / P * S2 / P) of floating values (range prob -1 ~ 1)
    heatmap combination requires using opencv (and therefore numpy arrays)
    '''

    if mask is None:
        img = rearrange(img.cpu().numpy(), 'c h w -> h w c')
        img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')
        result = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = Image.fromarray(result)
        return result


    if power:
        mask = (mask ** 4)

    # convert to numpy array
    mask = rearrange(mask, '(h w) -> h w', h=int(mask.shape[0] ** 0.5))
    mask = mask.detach().cpu().numpy()

    if color:
        mask = cv2.normalize(
            mask.astype('float32'), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        mask = mask.astype('uint8')

    mask = cv2.resize(mask, (img.shape[-1], img.shape[-1]))

    img = rearrange(img.cpu().numpy(), 'c h w -> h w c')
    img = cv2.cvtColor(img * 255, cv2.COLOR_RGB2BGR).astype('uint8')

    if color:
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    else:
        mask = rearrange(mask, 'h w -> h w 1')        

    if color:
        result = cv2.addWeighted(mask, 0.5, img, 0.5, 0)
    else:
        result = (mask * img)
        result = cv2.normalize(
            result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result = result.astype('uint8')

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    result = Image.fromarray(result)

    return result


def attention_rollout(scores_soft):
    # https://github.com/jeonsworld/ViT-pytorch/blob/main/visualize_attention_map.ipynb
    att_mat = torch.stack(scores_soft)

    # Average the attention weights across all heads.
    att_mat = reduce(att_mat, 'l b h s1 s2 -> l b s1 s2', 'mean')

    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(-1), device=att_mat.device)
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size(), device=att_mat.device)
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    return joint_attentions[-1]


def setup_environment(args):
    set_random_seed(args.seed, numpy=True)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    args.cal_ap_only = True
    model = build_model(args)

    if args.ckpt_path:
        args.setting = 'ft'
    elif args.pretrained:
        args.setting = 'pt'
    else:
        args.setting = 'scratch'
    args = adjust_args_general(args)

    if not args.debugging:
        wandb.init(config=args, project=args.project_name, entity=args.entity)
        wandb.run.name = args.run_name

    layers = get_layers(model, args.model_name, args.selector, args.vis_mask)

    hook = VisHook(model, layers, args.vis_mask, args.device)

    return train_loader, test_loader, hook


def get_layers(model, model_name, selector=None, vis_mask=None):
    if vis_mask is None:
        layers = []
        for name, _ in model.named_modules():
            layers.append(name)

    elif vis_mask in ('rollout', 'attention', 'gls') and any([kw in model_name for kw in ['vit', 'deit']]):
        # only works for deit/vit base
        layers = []
        for name, _ in model.named_modules():
            # print(name)
            if ('vit_b16' in model_name) and ('attn.drop' in name or 'encoder_norm' in name):
                layers.append(name)
            elif ('deit' in model_name or 'vit' in model_name) and ('attn_drop' in name or name == 'model.norm'):
                layers.append(name)

    elif vis_mask in ('rollout', 'attention'):
        raise NotImplementedError

    elif 'bap' in vis_mask and selector == 'cal':
        layers = ['model.dfsm.attentions.bn']

    elif 'bap' in vis_mask and selector != 'cal':
        raise NotImplementedError

    else:

        if model_name == 'vgg19_bn':
            layers = ['model.features.50']
        elif model_name == 'resnet18':
            layers = ['model.layer4.1.bn2']
        elif model_name == 'resnet34':
            layers = ['model.layer4.2.bn2']
        elif model_name == 'resnet50':
            layers = ['model.layer4.2.bn3']
        elif model_name == 'resnet101':
            layers = ['model.layer4.2.bn3']
        elif 'vit_b' in model_name:
            layers = ['model.encoder.blocks.11.norm2']
        elif 'beitv2_base_patch16_224_in22k' in model_name or 'deit3_base_patch16_224' in model_name:
            layers = ['model.blocks.11.norm2']
        elif 'deit3_large_patch16_224' in model_name:
            layers = ['model.blocks.23.norm2']
        elif model_name == 'van_b3':
            layers = ['model.norm4']
        elif 'convnext' in model_name:
            layers = ['model.stages.3.blocks.2.norm']
        elif 'convnext_base' in model_name:
            layers = ['model.stages.3.blocks.2.norm']
        elif 'swin' in model_name:
            layers = ['model.layers.3.blocks.1.norm2']
        elif 'swin_base' in model_name:
            layers = ['model.layers.3.blocks.1.norm2']
        elif 'resnetv2_101' in model_name:
            layers = ['model.stages.3.blocks.2.norm3']
        else:
            layers = []
            for name, _ in model.named_modules():
                # print(name)
                if 'norm' in name or 'bn' in name:
                    layers.append(name)
            layers = layers[-1]

        if selector == 'cal' and any([kw in model_name for kw in ('vit', 'deit', 'beit', 'swin', 'van')]):
            layers = [layer.replace('model.', 'model.encoder.0.') for layer in layers]
        elif selector == 'cal':
            layers = [layer.replace('model.', 'model.encoder.') for layer in layers]

    return layers


def main():
    args = parse_inference_args()

    amp_autocast = torch.cuda.amp.autocast if args.fp16 else suppress

    train_loader, test_loader, hook = setup_environment(args)
    set_random_seed(args.seed, numpy=True)

    # with torch.no_grad():
    with amp_autocast():
        hook.save_vis(args, train_loader, split='train')
        hook.save_vis(args, test_loader, split='test')

    if not args.debugging:
        wandb.finish()

    return 0


if __name__ == "__main__":
    main()
