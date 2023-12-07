import enum
import pathlib
import random
from functools import partial
from typing import List, Union, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import functional as F
from loguru import logger
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from torch.utils.data import Dataset

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


TensorCHW = Union[torch.Tensor, np.ndarray]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
UNIFORM_MEAN = (0.5, 0.5, 0.5)
UNIFORM_STD = (0.5, 0.5, 0.5)


class CustomDataset(Dataset):

    def __init__(self, phase, root_folder,
                 image_size: Union[int, Tuple[int, int]],
                 use_depth: bool = False,
                 smp_preprocessing: bool = False,
                 smp_encoder: Optional[str] = None,
                 edge_kernel: int = 3,
                 imagenet_norm: bool = False,
                 ):
        assert phase in ('training', 'validation', 'testing')
        self.root = pathlib.Path(root_folder)
        self.phase = phase
        # self.metadata = load_lsun_mat(self.root / f'{phase}.mat')
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.target_size = image_size
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2,
                                          saturation=0.2)
        self.subdir = 'train' if phase == 'training' else 'val'
        self.use_depth = use_depth
        self.edge_kernel = edge_kernel

        if smp_preprocessing:
            logger.info('Use SMP preprocessing')
            assert smp_encoder is not None, 'Expected not None encoder name'
            self._preprocess = partial(
                self._preprocess_smp,
                get_preprocessing_fn(smp_encoder))
        else:
            self._preprocess = self._preprocess_common

        self.img_fpaths = list(
            (self.root / self.subdir / 'images')
            .glob('*[.jpg|.jpeg|.png|.webp|.lfif]')
        )
        self.norm_mean = IMAGENET_MEAN if imagenet_norm else UNIFORM_MEAN
        self.norm_std = IMAGENET_STD if imagenet_norm else UNIFORM_STD

    def _preprocess_common(self, image: Image.Image) -> torch.Tensor:
        image = F.to_tensor(image)
        image = F.normalize(image, mean=self.norm_mean, std=self.norm_std)
        return image

    def _preprocess_smp(
            self, smp_preproc: callable, image: Image.Image
    ) -> torch.Tensor:
        image = np.array(image)
        image = smp_preproc(image)
        image = F.to_tensor(image).float()
        return image

    def load_item(self, img_fpath: pathlib.Path):
        fname = img_fpath.stem
        image_path = img_fpath
        label_path = self.root / self.subdir / 'masks' / f'{fname}.png'
        if self.use_depth:
            depth_path = self.root / self.subdir / 'depth' / f'{fname}.png'

        image = Image.open(image_path).convert('RGB')
        image = self._preprocess(image)

        label = torch.from_numpy(cv2.imread(str(label_path),
                                            cv2.IMREAD_UNCHANGED))[None]
        assert label is not None, f'Label is None: {label_path}'
        if self.use_depth:
            depth = F.to_tensor(Image.open(depth_path).convert('L'))

        if self.phase == 'training':
            image = self.color_jitter(image)
            image, label = random_h_flip(image, label)
            # Don't use because of unknown layout_type for new images
            # image, label, layout_type = random_layout_degradation(
            #     image, label, layout_type)

        image = F.resize(image, self.target_size, Image.BILINEAR)
        if self.use_depth:
            depth = F.resize(depth, self.target_size, Image.BILINEAR)
            image = torch.cat((image, depth), dim=0)
        label = F.resize(label, self.target_size, Image.NEAREST)
        edge_map = generate_edge_map_from(label[0].numpy(), k=self.edge_kernel)

        room_max = torch.max(label)
        if room_max > Layout.wall7.value:
            raise RuntimeError(f'Found surface with label {room_max}. '
                               f'The expected max value is {Layout.wall7}')
        item = {
            'image': image,
            # make 0 into 255 as ignore index
            'label': label[0].clamp_(0, Layout.get_max_wall_value()).long(),
            'edge': torch.from_numpy(edge_map).clamp_(0, 1).float(),
            # 'type': layout_type,
        }
        return item

    def __len__(self):
        return len(self.img_fpaths)

    def __getitem__(self, index):
        return self.load_item(self.img_fpaths[index])

    def to_loader(self, batch_size, num_workers=0):
        return torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=self.phase == 'training',
            pin_memory=True, num_workers=num_workers
        )


def denormalize_image(
        image: torch.Tensor, mean: List[float], std: List[float]
) -> torch.Tensor:
    assert len(std) == 3 and len(mean) == 3, ('Expected mean and std to have '
                                              '3 elements')
    inv_transform = T.Compose(
        [T.Normalize(mean=[0., 0., 0.],
                     std=[1 / std[0], 1 / std[1], 1 / std[2]]),
         T.Normalize(mean=[-mean[0], -mean[1], -mean[2]], std=[1., 1., 1.])]
    )
    return inv_transform(image)


def generate_edge_map_from(label, k=5):
    lbl = cv2.GaussianBlur(label.astype('uint8'), (3, 3), 0)
    edge = cv2.Laplacian(lbl, cv2.CV_64F)
    activation = cv2.dilate(np.abs(edge), np.ones((k, k), np.uint8),
                            iterations=1)
    activation[activation != 0] = 1
    # TODO: Maybe modify kernel if this blur
    return cv2.GaussianBlur(activation, (15, 15), 5)


def random_h_flip(image: TensorCHW, label: TensorCHW):
    if 0.5 < torch.rand(1):
        return image, label

    image = T.functional.hflip(image)
    label = T.functional.hflip(label)

    label = label.numpy()
    old_label = label.copy()

    room_max = np.max(label)
    for i in range(Layout.wall1.value, room_max + 1):
        new_room_label = Layout.wall1.value + (room_max - i)
        logger.debug(f'{i} -> {new_room_label}')
        label[old_label == i] = new_room_label
    label = torch.from_numpy(label)

    return image, label


def accept_aspect_ratio(x: TensorCHW):
    try:
        h, w = x.shape[1:]
        ratio = h / w if h > w else w / h
        return ratio < 16 / 9
    except (ZeroDivisionError, RuntimeWarning):
        return False


class Layout(enum.Enum):
    ceil = 1
    floor = 2
    wall1 = 3
    wall2 = 4
    wall3 = 5
    wall4 = 6
    wall5 = 7
    wall6 = 8
    wall7 = 9

    @staticmethod
    def get_max_wall_value():
        return Layout.wall7.value


def _get_colors():
    random.seed(1)
    colors = [
        tuple([random.randint(0, 255) for _ in range(3)])
        for _ in range(Layout.get_max_wall_value() - 2)
    ]
    colors = [(255, 255, 255), (128, 128, 128), *colors]
    return colors


COLORS = _get_colors()


def get_colorized_layout(mask: np.ndarray):
    assert mask.ndim == 2
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, color in enumerate(COLORS):
        label = i + 1
        color_mask[mask == label] = color
    return color_mask


def vis_dataset(phase, folder, image_size, use_depth=False,
                smp_preprocessing=False, smp_encoder=None,
                edge_kernel=3) -> None:
    dataset = CustomDataset(
        phase, folder, image_size,
        use_depth=use_depth,
        smp_preprocessing=smp_preprocessing,
        smp_encoder=smp_encoder,
        edge_kernel=edge_kernel
    )
