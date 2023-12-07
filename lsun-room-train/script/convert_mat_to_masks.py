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


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
        mat_dir: Path,
        masks_dir: Path,
        overwrite: bool = False
):
    if not masks_dir.exists():
        masks_dir.mkdir(parents=True)
    elif not overwrite:
        logger.error(f'Directory {masks_dir} already exists.')
        return

    for mat_fpath in tqdm(list(mat_dir.glob('*.mat'))):
        layout = loadmat(str(mat_fpath))['layout']
        dst_fpath = masks_dir / f'{mat_fpath.stem}.png'
        cv2.imwrite(str(dst_fpath), layout.astype(np.uint8))


if __name__ == '__main__':
    app()
