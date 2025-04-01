import os
import time
import random

import wandb
from timm.optim import create_optimizer
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from fgir_kd.data_utils.build_dataloaders import build_dataloaders
from fgir_kd.model_utils.build_model import build_model
from fgir_kd.other_utils.build_args import parse_train_args
from fgir_kd.train_utils.misc_utils import summary_stats, stats_test, set_random_seed, count_flops
from fgir_kd.train_utils.scheduler import build_scheduler
from fgir_kd.train_utils.trainer import Trainer
from fgir_kd.train_utils.calc_loss import OverallLoss


IGNORE = (
    'project_name', 'ckpt_path', 'transfer_learning',
    'test_only', 'test_multiple', 'offline', 'debugging', 'distributed',
    'batch_size', 'lr', 'epochs', 'eval_freq', 'cpu_workers',
    'vis_errors', 'vis_errors_save',
    'transfer_learning_cal', 'top_k',
)


def adjust_args_general(args):
    selector = f'_{args.selector}' if args.selector else ''

    args.run_name = '{}_{}{}_{}'.format(
        args.dataset_name, args.model_name, selector, args.serial
    )

    args.results_dir = os.path.join(args.results_dir, args.run_name)

    return args


def build_environment(args):
    # if args.ckpt_path and not args.transfer_learning:
    #     args_temp = vars(torch.load(args.ckpt_path, map_location=torch.device('cpu'))['config'])
    #     for k, v in args_temp.items():
    #         if k not in IGNORE:
    #             if ((k == 'dataset_root_path' and getattr(args, k, None) is not None) or
    #                 (k == 'test_resize_size' and (
    #                     getattr(args, k) >= args_temp['image_size'] and
    #                     getattr(args, k) < v))):
    #                 pass
    #             else:
    #                 setattr(args, k, v)

    if args.serial is None:
        args.serial = random.randint(0, 1000)
    # Set device and random seed
    set_random_seed(args.seed, numpy=False)

    # dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(args)

    # model and criterion
    model = build_model(args)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    model.zero_grad()

    criterion = OverallLoss(args)

    # loss and optimizer
    optimizer = create_optimizer(args, model)
    lr_scheduler = build_scheduler(args, optimizer, train_loader)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    # if not args.ckpt_path or args.transfer_learning:
    adjust_args_general(args)
    os.makedirs(args.results_dir, exist_ok=True)

    return model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader


def main():
    time_start = time.time()

    args = parse_train_args()

    model, criterion, optimizer, lr_scheduler, train_loader, val_loader, test_loader = build_environment(args)

    trainer = Trainer(args, model, criterion, optimizer, lr_scheduler,
                      train_loader, val_loader, test_loader)

    flops = count_flops(model, args.image_size, args.device, 'torchprofile')

    if args.set_seed_all_before_train_loop:
        set_random_seed(args.seed, numpy=True)

    if args.test_only:
        if not args.vis_errors and not args.debugging and not args.offline:
            wandb.init(config=args, project=args.project_name, entity=args.entity)
            wandb.run.name = args.run_name
        time_start = time.time()
        print(args, model.cfg)

        test_acc, max_memory, no_params, no_params_trainable, class_deviation = trainer.test()

        if not args.debugging:
            time_total = time.time() - time_start

            if args.test_multiple:
                num_images = (args.test_multiple + 1) * args.num_images_test
            else:
                num_images = args.num_images_test

            stats_test(test_acc, class_deviation, flops, max_memory, no_params, no_params_trainable,
                       time_total, num_images, (args.vis_errors or args.offline))
            if not args.vis_errors and not args.offline:
                wandb.finish()
    else:
        if args.local_rank == 0:
            if not args.debugging and not args.offline:
                wandb.init(config=args, project=args.project_name, entity=args.entity)
                wandb.run.name = args.run_name
            if not args.distributed:
                print(model)
            print(args)

        best_acc, best_epoch, max_memory, no_params, no_params_trainable, class_deviation = trainer.train()

        # summary stats
        if args.local_rank == 0 and not args.debugging:
            time_total = time.time() - time_start
            summary_stats(args.epochs, time_total, best_acc, best_epoch, flops, max_memory,
                          no_params, no_params_trainable, class_deviation, args.offline)
            if not args.offline:
                wandb.finish()


if __name__ == '__main__':
    main()
