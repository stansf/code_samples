import numpy as np
import torch
from loguru import logger

from .trainer import core

torch.backends.cudnn.benchmark = True


def blend(image: torch.Tensor, outputs: torch.Tensor,
          alpha: float = 0.4) -> np.ndarray:
    label = core.label_as_rgb_visual(outputs.unsqueeze(0)).squeeze(0)
    blend_output = (image / 2 + .5) * (1 - alpha) + (label * alpha)
    blend_output_np = blend_output.permute(1, 2, 0).numpy()
    return np.clip(blend_output_np * 255, 0, 255).astype(np.uint8)


def combine_classes(outputs: torch.Tensor) -> torch.Tensor:
    #     frontal = 1
    #     left = 2
    #     right = 3
    #     floor = 4
    #     ceiling = 5
    outputs = outputs.clone()
    mask_wall = torch.isin(outputs, torch.as_tensor((0, 1, 2)))
    outputs[mask_wall] = 0
    outputs[outputs == 3] = 3
    outputs[outputs == 4] = 4
    return outputs


class Predictor:
    def __init__(self, weight_path, device):
        self.model = core.LayoutSeg.load_from_checkpoint(
            weight_path, backbone='resnet101'
        )
        self.model.freeze()
        self.model.to(device)
        logger.info('Room layout: model weights loaded.')

    @torch.no_grad()
    def feed(self, image: torch.Tensor) -> np.ndarray:
        _, outputs = self.model(image.unsqueeze(0).cuda())
        outputs = outputs.squeeze(0).cpu()
        # outputs = combine_classes(outputs)
        outputs = outputs.numpy()
        logger.debug('Room layout: make inference')
        return outputs

    @torch.no_grad()
    def feed_with_blend(self, image: torch.Tensor, alpha=.4,
                        less_classes=False) -> np.ndarray:
        _, outputs = self.model(image.unsqueeze(0).cuda())
        outputs = outputs.squeeze(0).cpu()
        if less_classes:
            outputs = combine_classes(outputs)
        logger.debug(f'Room layout: Make inference with blending, '
                     f'less classes: {less_classes}')
        return blend(image, outputs, alpha=alpha)
