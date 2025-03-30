import os
import glob

import pandas as pd
from PIL import Image
import torch
# from einops import rearrange
# import torch.nn.functional as F
# from torchvision.utils import save_image
# import timm

from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.data_utils.build_transform import build_transform
# from fgir_kd.train_utils.save_vis_images import inverse_normalize

# from train import build_environment
# from heatmap import make_heatmaps, inverse_normalize
from vis_dfsm import build_environment_inference


def prepare_img(fn, args, transform):
    # open img
    img = Image.open(fn).convert('RGB')
    # Preprocess image
    img = transform(img).unsqueeze(0).to(args.device)
    return img


def search_images(folder):
    # the tuple of file types
    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')

    # if folder is a file
    if os.path.isfile(folder):
        # if folder is a .txt or .csv with file names
        if os.path.splitext(folder)[1] in ('.txt', '.csv'):
            df = pd.read_csv(folder)
            print('Total image files', len(df))
            return df['dir'].tolist()

        # if folder is a path to an image
        elif any([t.replace('*', '') in os.path.splitext(folder)[1] for t in types]):
            return [folder]

    # else if directory
    files_all = []
    for file_type in types:
        # files_all is the list of files
        path = os.path.join(folder, '**', file_type)
        files_curr_type = glob.glob(path, recursive=True)
        files_all.extend(files_curr_type)

        print(file_type, len(files_curr_type))

    print('Total image files', len(files_all))
    return files_all


# def adjust_args_general(args):
#     ila = '_ila' if args.ila else ''
#     if ila:
#         ila = ila if args.ila_locs else f'{ila}_dso'
#     classifier = f'_{args.classifier}' if args.classifier else ''
#     selector = f'_{args.selector}' if args.selector else ''
#     adapter = f'_{args.adapter}' if args.adapter else ''
#     prompt = f'_{args.prompt}' if args.prompt else ''
#     freeze = '_fz' if args.freeze_backbone else ''

#     args.run_name = '{}_{}{}{}{}{}{}{}_{}'.format(
#         args.dataset_name, args.model_name, ila, classifier, selector, adapter,
#         prompt, freeze, args.serial
#     )

#     args.results_dir = os.path.join(args.results_inference, args.run_name)
#     os.makedirs(args.results_dir, exist_ok=True)

#     return args

# def setup_environment(args):
#     set_random_seed(args.seed, numpy=True)

#     _, _, _ = build_dataloaders(args)

#     model = build_model(args)
#     model.eval()

#    return model

def prepare_inference(args):

    # model, _, _, _, _, _, _ = build_environment(args)
    # model.eval()

    # if args.results_inference:
    #     if args.dynamic_anchor:
    #         model_name = f'{args.model_name}_{args.selector}'
    #     else:
    #         model_name = f'{args.model_name}'
    #     args.results_dir = os.path.join(args.results_inference, f'{model_name}')
    #     os.makedirs(args.results_dir, exist_ok=True)

    # model = setup_environment(args)

    _, _, hook, amp_autocast = build_environment_inference(args)

    transform = build_transform(args=args, split='test')

    # Load class names
    dic_classid_classname = None

    if args.dataset_root_path and args.df_classid_classname:
        fp = os.path.join(args.dataset_root_path, args.df_classid_classname)

        if os.path.isfile(fp):
            dic_classid_classname = pd.read_csv(fp, index_col='class_id')['class_name'].to_dict()
    
    return hook, amp_autocast, transform, dic_classid_classname


# def save_crops(images_og, images_crops, fp, image_size,
#                 save_crops_only=False, norm_custom=False):
#     fp = fp.replace('.png', '_crops.png')

#     with torch.no_grad():
#         if images_crops is not None:
#             images_crops = images_crops.reshape(3, image_size, -1)

#         if save_crops_only and images_crops is not None:
#             samples = inverse_normalize(images_crops.data, norm_custom)
#         else:
#             images_og = images_og.reshape(3, image_size, image_size)

#             if images_crops is not None:
#                 images = torch.cat((images_og, images_crops), dim=2)
#             else:
#                 images = images_og
#             samples = inverse_normalize(images.data, norm_custom)

#         save_image(samples, fp, nrow=1)
#         print(f'Saved file to : {fp}')
#     return 0


# def inference_single(args, amp_autocast, hook, img, dic_classid_classname=None,
#                      file=None, save=False, images_crops=None, scores=None, scores_soft=None, x_norm=None, 
#                      inter=None, fp=None, masked_image=None):
def inference_single(args, amp_autocast, hook, img, dic_classid_classname=None,
                     file=None, save=False):

    fn = os.path.splitext(os.path.split(file)[1])[0]
    # fp = os.path.join(args.results_dir, fn)
    # save_crops(img.detach().clone(), images_crops, fp, args.image_size,
    #             args.save_crops_only, args.custom_mean_std)

    # with torch.no_grad():
    #     outputs = model(img, ret_dist=True)

    # if args.model_name == 'cal':
    #     outputs, images_crops = outputs
    # elif 'van' in args.model_name or args.model_name in timm.list_models():
    #     outputs, inter = outputs
    # elif isinstance(outputs, tuple) and len(outputs) == 6:
    #     outputs, images_crops, x_norm, inter, scores_soft, scores = outputs
    # elif isinstance(outputs, tuple) and len(outputs) == 5:
    #     outputs, x_norm, inter, scores_soft, scores = outputs
    # elif isinstance(outputs, tuple) and len(outputs) == 3:
    #     outputs, scores_soft, scores = outputs
    # elif isinstance(outputs, tuple) and len(outputs) == 2:
    #     outputs, _ = outputs

    with amp_autocast():
        if save:
            preds, pil_image = hook.inference_save_vis(args, img, save_name=fn)
        else:
            preds, _ = hook.inference(img)

    preds = preds.squeeze(0)
    for i, idx in enumerate(torch.topk(preds, k=3).indices.tolist()):
        prob = torch.softmax(preds, -1)[idx].item()
        if dic_classid_classname is not None:
            classname = dic_classid_classname[idx]
            out_text = '[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=classname, p=prob*100)
            print(out_text)
        else:
            out_text = '[{idx}] ({p:.2f}%)'.format(idx=idx, p=prob*100)
            print(out_text)
        if i == 0:
            top1_text = out_text

    # if save:
    #     if args.model_name == 'cal':
    #         if args.cal_save_all:
    #             images_crops = rearrange(images_crops, 'b c h w k -> b c h (k w)')
    #         else:
    #             images_crops = images_crops[:, :, :, :, 0]

    #     fn = '{}.png'.format(os.path.splitext(os.path.split(file)[1])[0])
    #     fp = os.path.join(args.results_dir, fn)
    #     save_crops(img.detach().clone(), images_crops, fp, args.image_size,
    #                args.save_crops_only, args.custom_mean_std)

    # if (scores is not None or inter is not None) and (args.vis_mask or args.vis_mask_all):
    #     masked_image = make_heatmaps(args, fp, img, model, inter, scores_soft,
    #                                     scores, x_norm, args.vis_mask_all, save=save)

    if save:
        return top1_text, pil_image
    return top1_text
    # return top1_text, images_crops, masked_image


def inference_all(args):
    files_all = search_images(args.images_path)

    hook, amp_autocast, transform, dic_classid_classname = prepare_inference(args)

    for file in files_all:
        print(file)
        img = prepare_img(file, args, transform)

        # Classify
        inference_single(args, amp_autocast, hook, img, dic_classid_classname,
                         file, save=args.vis_mask)

        # if args.debugging:
        #     print('Finished.')
        #     return 0

    return 0


def main():
    args = parse_inference_args()
    args.debugging = True

    inference_all(args)

    return 0


if __name__ == '__main__':
    main()
