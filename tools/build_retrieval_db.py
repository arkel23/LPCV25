import os
from contextlib import suppress

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageFile

import torch
from torch.nn import functional as F
from torchvision import transforms
import torch.utils.data as data
import timm


from fgir_kd.other_utils.build_args import parse_inference_args
from fgir_kd.train_utils.misc_utils import set_random_seed


ImageFile.LOAD_TRUNCATED_IMAGES = True


def adjust_args_general(args):
    args.run_name = '{}_{}_{}'.format(
        args.dataset_name, args.model_name, args.serial
    )

    args.results_dir = os.path.join(args.results_inference, args.run_name)

    return args


def get_set(args, split, transform=None):
    ds = DatasetImgTargetDir(args, split=split, transform=transform)
    args.num_classes = ds.num_classes

    setattr(args, f'num_images_{split}', ds.__len__())
    print(f"{args.dataset_name} {split} split. N={ds.__len__()}, K={ds.num_classes}.")
    return ds


class DatasetImgTargetDir(data.Dataset):
    def __init__(self, args, split, transform=None):
        self.root = os.path.abspath(args.dataset_root_path)
        self.transform = transform
        self.dataset_name = args.dataset_name

        if split == 'train':
            if args.train_trainval:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_trainval
            else:
                self.images_folder = args.folder_train
                self.df_file_name = args.df_train
        elif split == 'val':
            if args.train_trainval:
                self.images_folder = args.folder_test
                self.df_file_name = args.df_test
            else:
                self.images_folder = args.folder_val
                self.df_file_name = args.df_val
        else:
            self.images_folder = args.folder_test
            self.df_file_name = args.df_test

        assert os.path.isfile(os.path.join(self.root, self.df_file_name)), \
            f'{os.path.join(self.root, self.df_file_name)} is not a file.'

        self.df = pd.read_csv(os.path.join(self.root, self.df_file_name), sep=',')
        self.targets = self.df['class_id'].to_numpy()
        self.data = self.df['dir'].to_numpy()

        self.num_classes = len(np.unique(self.targets))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_dir, target = self.data[idx], self.targets[idx]
        full_img_dir = os.path.join(self.root, self.images_folder, img_dir)
        img = Image.open(full_img_dir)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, target, img_dir

    def __len__(self):
        return len(self.targets)


class ImitateQAI:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, img):
        img = img.resize((self.image_size, self.image_size))
        img = np.array(img, dtype=np.float32) / 255.0
        return img


def build_transform(args):
    image_size = args.image_size

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]

    t = []
    t.append(ImitateQAI(image_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=mean, std=std))
    transform = transforms.Compose(t)

    return transform


def get_loader(args, split):
    transform = build_transform(args=args)
    ds = get_set(args, split, transform)

    shuffle = True if split == 'train' else False

    data_loader = data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=shuffle,
        num_workers=args.cpu_workers, drop_last=False)

    return data_loader


def build_dataloaders(args):
    train_loader = get_loader(args, 'train')
    test_loader = get_loader(args, 'test')
    return train_loader, test_loader


# Custom wrapper class for preprocessing and model
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model_name='regnety_040', img_size=224):
        super(FeatureExtractor, self).__init__()
        # Load model with the specified number of classes

        if 'vit' in model_name or 'deit' in model_name:
            classifier = 'cls'
            model = timm.create_model(
                model_name,
                num_classes=0, 
                img_size=img_size,
                global_pool='',
            )
        else:
            classifier = 'gap'
            model = timm.create_model(
                model_name,
                num_classes=0, 
                global_pool=''
            )

        s, d, bsd = self.get_out_features(img_size, model)
        self.output_channels = d

        self.model = model


    @torch.no_grad()
    def get_out_features(self, image_size, model):
        x = torch.rand(2, 3, image_size, image_size)
        x = model(x)

        if len(x.shape) == 3:
            b, s, d = x.shape
            bsd = True
        elif len(x.shape) == 4:
            b, d, h, w = x.shape
            s = h * w
            bsd = False

        print('Output feature shape: ', x.shape)

        return s, d, bsd

    def forward(self, img):
        # Pass the preprocessed image through the model
        features = self.model(img)

        b, d, _, _ = features.shape
        features = torch.mean(features.view(b, d, -1), dim=-1)

        features = F.normalize(features, dim=-1)

        return features


def extract_features(model, loader,split, args):
    features_all = torch.empty((0, args.output_channels), device=args.device)
    targets_all = torch.empty((0,), device=args.device, dtype=torch.long)
    dirs_all = []
    images_per_class = {k: 0 for k in range(0, args.num_classes)}

    # Use fp16 calculation
    amp_autocast = torch.amp.autocast if args.fp16 else suppress
    with torch.no_grad():
        with amp_autocast(args.device):
            for idx, (images, targets, dirs) in enumerate(loader):
                images = images.to(args.device, non_blocking=True)
                targets = targets.to(args.device, non_blocking=True)

                features = model(images)


                if args.db_images_per_class:
                    for (i, dir) in enumerate(dirs):
                        feature = features[i:i+1]
                        target = targets[i:i+1]

                        if images_per_class[int(target)] >= args.db_images_per_class:
                            continue

                        images_per_class[int(target)] += 1
                        features_all = torch.cat([features_all, feature])
                        targets_all = torch.cat([targets_all, target])
                        dirs_all.append(dir)


                elif args.db_size:
                    features_all = torch.cat([features_all, features])
                    targets_all = torch.cat([targets_all, targets])
                    dirs_all.extend(dirs)

                    for target in targets:
                        images_per_class[int(target)] += 1

                    if len(features_all) >= args.db_size:
                        print('Reached minimum number of images')
                        break


                if idx % args.log_freq == 0:
                    print(f"Extracting {split} features: {idx} / {len(loader)}")


    db = {
        'features': features_all,
        'classes': targets_all,
        'dir': dirs_all,
        'images_per_class': images_per_class,
    }

    args.db_size = len(features_all)

    fp = os.path.join(args.results_dir, f'db_{split}_{args.db_size}.pth')
    torch.save(db, fp)
    print(f'Saved retrieval DB to {fp}')

    return 0


def main():
    # input args and constants
    args = parse_inference_args()
    # args.db_size = 1000
    # args.db_images_per_class = 2

    # Set device and random seed
    set_random_seed(args.seed, numpy=False)

    adjust_args_general(args)
    os.makedirs(args.results_dir, exist_ok=True)

    # dataloaders
    train_loader, test_loader = build_dataloaders(args)

    # Create the model
    model = FeatureExtractor(args.model_name, args.image_size)
    model.eval()
    model.to(args.device)
    args.output_channels = model.output_channels

    # load checkpoint
    if args.ckpt_path:
        state_dict = torch.load(args.ckpt_path, map_location=torch.device('cpu'))['model']
        expected_missing_keys = []
        ret = model.load_state_dict(state_dict, strict=False)
        print('''Missing keys when loading pretrained weights: {}
                Expected missing keys: {}'''.format(ret.missing_keys, expected_missing_keys))
        print('Unexpected keys when loading pretrained weights: {}'.format(
            ret.unexpected_keys))
        print('Loaded from custom checkpoint.')

    for (split, loader) in zip(['train', 'test'], [train_loader, test_loader]):
        extract_features(model, loader, split, args)

    return 0


if __name__ == '__main__':
    main()

