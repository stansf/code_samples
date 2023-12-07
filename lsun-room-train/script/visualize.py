from pathlib import Path
from pycocotools.coco import COCO
from matplotlib import pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import k_means
from loguru import logger
from typing import List, Optional
from tqdm import tqdm
import typer
from scipy.io import loadmat, savemat

# RGB
COLOR_MAPPING = {
    2: [255, 0, 0],  # Left
    1: [0, 255, 0],  # Front
    3: [0, 0, 255],  # Right
    4: [255, 255, 0],  # Ceil
    5: [255, 0, 255],  # Floor
}


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
        src_masks_dir: Path,
        vis_dir: Path,
        imgs_dir: Optional[Path] = None,
        overwrite: bool = False
):
    if not vis_dir.exists():
        vis_dir.mkdir(parents=True)
    elif not overwrite:
        logger.error(f'Directory {vis_dir} already exists.')
        return

    if imgs_dir:
        imgs_fpaths = {fpath.stem: fpath for fpath
                       in imgs_dir.glob('*[.jpg|.jpeg|.png|.webp|.jfif]')}

    for i, mask_fpath in enumerate(tqdm(sorted(list(
            src_masks_dir.glob('*.png'))))):
        src_mask = cv2.imread(str(mask_fpath), cv2.IMREAD_UNCHANGED)
        new_mask = np.zeros(src_mask.shape + (3,), dtype=np.uint8)
        for label in np.unique(src_mask):
            new_mask[src_mask == label] = COLOR_MAPPING[label][::-1]

        if imgs_dir:
            # img_fpath = imgs_dir /
            img_fpath = imgs_fpaths.get(mask_fpath.stem)
            if not img_fpath:
                logger.warning(f'Not found mask for mask {mask_fpath.name}')
                continue
            img = cv2.imread(str(img_fpath))
            try:
                new_mask = cv2.addWeighted(img, 0.8, new_mask, 0.3, 1)
            except cv2.error:
                logger.error(new_mask.shape)
                logger.error(img.shape)
                logger.error(mask_fpath.name)
                logger.error(imgs_fpaths[i].name)
                raise
        cv2.imwrite(str(vis_dir / mask_fpath.name), new_mask)


if __name__ == '__main__':
    app()
