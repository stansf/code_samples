import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Union, Tuple

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from loguru import logger

from trainer import core
from datasets.custom import CustomDataset, Layout


def parse_args():
    parser = argparse.ArgumentParser(
        description='Indoor room corner detection')
    parser.add_argument('--name', help='experiment name')
    parser.add_argument('--folder', default='data/lsun_room',
                        help='where is the dataset')
    parser.add_argument('--phase', default='eval',
                        choices=['train', 'eval', 'export'])
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--ch', default=3, type=int)
    parser.add_argument('--log_freq_img', default=100, type=int)

    # data
    parser.add_argument('--image_size', default=320, type=int)
    parser.add_argument('--use_layout_degradation', action='store_true')
    parser.add_argument('--edge_kernel', default=15, type=int,
                        help='Kernel of dilation of edges. Should be less on '
                             'later epochs. E.g 15 -> 5')
    parser.add_argument('--imagenet_norm', action='store_true',
                        help='Use ImageNet mean and std')
    parser.add_argument('--smp_preprocessing', action='store_true')

    # network
    parser.add_argument('--arch', default='resnet')
    parser.add_argument('--backbone', default='resnet101')
    parser.add_argument('--optim', default='adam')
    parser.add_argument('--pretrain_path', default='')
    parser.add_argument('--model_type', type=core.ModelTypes,
                        default='original')
    parser.add_argument('--pretrained_backbone',
                        action='store_true')
    parser.add_argument('--smp_arch', type=str, default='unet')

    # hyper-parameters
    parser.add_argument('--l1_factor', type=float, default=0.0)
    parser.add_argument('--l2_factor', type=float, default=0.0)
    parser.add_argument('--edge_factor', type=float, default=0.0)

    parser.add_argument('--onnx_path', type=str)
    args = parser.parse_args()
    return args


def create_dataloaders(
        dataset_root_dir: Union[Path, str],
        channels: int,
        image_size: Union[int, Tuple[int, int]],
        backbone: str,
        edge_kernel: int,
        batch_size: int,
        workers: int,
        imagenet_norm: bool,
        smp_preprocessing: bool,

):
    assert channels in (3, 4), f'Unknown num channels: {channels}'
    use_depth = channels == 4
    logger.info(f'Use depth: {use_depth}')
    train_dataset = CustomDataset(
        'training', root_folder=dataset_root_dir,
        image_size=image_size, use_depth=use_depth, smp_encoder=backbone,
        edge_kernel=edge_kernel, imagenet_norm=imagenet_norm,
        smp_preprocessing=smp_preprocessing
    )
    val_dataset = CustomDataset(
        'validation', root_folder=dataset_root_dir,
        image_size=image_size, use_depth=use_depth, smp_encoder=backbone,
        edge_kernel=edge_kernel, imagenet_norm=imagenet_norm,
        smp_preprocessing=smp_preprocessing
    )

    return (
        train_dataset.to_loader(batch_size=batch_size, num_workers=workers),
        val_dataset.to_loader(batch_size=1, num_workers=workers),
    )


def main():
    args = parse_args()
    logger.info(args)

    if args.phase == 'export':
        if pl.__version__ < '1.6.0':
            logger.warning('May need updated pytorch-lightning '
                           'to version >= 1.6')
        model = core.LayoutSeg.load_from_checkpoint(
            args.pretrain_path,
            backbone=args.backbone,
            model_type=args.model_type,
            smp_arch=args.smp_arch,
            input_ch=args.ch,
        )
        x = torch.rand((1, 3, args.image_size, args.image_size))
        model.to_onnx(args.onnx_path, x)
        return

    train_loader, val_loader = create_dataloaders(
        args.folder,
        args.ch, args.image_size, args.backbone, args.edge_kernel,
        args.batch_size, args.workers, args.imagenet_norm,
        args.smp_preprocessing
    )
    if args.phase == 'train':
        time_prefix = dt.datetime.now().strftime('%y-%m-%d_%H%M%S')
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f'ckpts/{args.name}/{time_prefix}',
            filename=f'{args.backbone}_'
                     + '{step}-{val_loss:.6f}',
            save_top_k=3, monitor='val_loss',
        )
        model = core.LayoutSeg(
            backbone=args.backbone,
            num_classes=Layout.get_max_wall_value() + 1,
            lr=args.lr,
            l1_factor=args.l1_factor, l2_factor=args.l2_factor,
            edge_factor=args.edge_factor, input_ch=args.ch,
            pretrained_backbone=args.pretrained_backbone,
            model_type=args.model_type,
            smp_arch=args.smp_arch,
            log_freq_img=args.log_freq_img
        )
        trainer = pl.Trainer(
            gpus=1,
            max_epochs=args.epochs,
            resume_from_checkpoint=args.pretrain_path or None,
            callbacks=[checkpoint_callback],
            logger=pl_loggers.TensorBoardLogger('ckpts/logs',
                                                name=args.name,
                                                version=time_prefix),
            accumulate_grad_batches=2,
        )
        trainer.fit(model, train_loader, val_loader)
    elif args.phase == 'eval':
        model = core.LayoutSeg.load_from_checkpoint(
            args.pretrain_path,
            backbone=args.backbone,
            model_type=args.model_type,
            smp_arch=args.smp_arch,
            input_ch=args.ch,
        )
        trainer = pl.Trainer(gpus=1, logger=False)
        result = trainer.test(model, val_loader)
        logger.info(f'Validate score: {result[0]["score"]}')


if __name__ == '__main__':
    logger.configure(handlers=[{"sink": sys.stdout, "level": "INFO"}])
    main()
