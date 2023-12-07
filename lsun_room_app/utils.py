from typing import Dict

import numpy as np


def label_as_rgb_visual_np(x: np.ndarray) -> np.ndarray:
    """
    Make segment tensor into colorful image.

    Args:
    ----
        x (np.ndarray): shape in (N, H, W) or (N, 1, H, W)

    Returns:
    -------
        canvas (np.ndarray): colorized tensor in the shape of (N, C, H, W).
    """
    colors = [
        [0.9764706, 0.27058825, 0.3647059],
        [1., 0.8980392, 0.6666667],
        [0.5647059, 0.80784315, 0.70980394],
        [0.31764707, 0.31764707, 0.46666667],
        [0.94509804, 0.96862745, 0.8235294]
    ]

    if x.ndim == 4:
        x = x.squeeze(1)
    assert x.ndim == 3

    n, h, w = x.shape
    palette = np.array(colors)
    labels = np.arange(x.max() + 1)

    canvas = np.zeros((n, h, w, 3))
    for color, lbl_id in zip(palette, labels):
        if canvas[x == lbl_id].shape[0]:
            canvas[x == lbl_id] = color

    return np.transpose(canvas, (0, 3, 1, 2))


def blend_np(image: np.ndarray, outputs: np.ndarray,
             alpha: float = 0.4) -> np.ndarray:
    label = label_as_rgb_visual_np(outputs[None, ...]).squeeze(0)
    blend_output = (
            (image / 2 + .5) * (1 - alpha) + (label * alpha)
    ).squeeze(0)
    blend_output_np = np.transpose(blend_output, (1, 2, 0))
    return np.clip(blend_output_np * 255, 0, 255).astype(np.uint8)


def get_classes() -> Dict[str, int]:
    return {
        'frontal': 0,
        'left': 1,
        'right': 2,
        'floor': 3,
        'ceiling': 4
    }
