from pathlib import Path

import numpy as np
import cv2
from loguru import logger
from tqdm import tqdm
import typer

from datasets.custom import Layout as ALayout
from datasets.lsunroom import Layout as LLayout


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
        src_masks_dir: Path,
        dst_masks_dir: Path,
        overwrite: bool = False
):
    if not dst_masks_dir.exists():
        dst_masks_dir.mkdir(parents=True)
    elif not overwrite:
        logger.error(f'Directory {dst_masks_dir} already exists. Cancel.')
        return

    for fpath in tqdm(list(src_masks_dir.glob('*.png'))):
        src_mask = cv2.imread(str(fpath), cv2.IMREAD_UNCHANGED)
        new_mask = src_mask.copy()
        new_mask[src_mask == LLayout.floor.value] = ALayout.floor.value
        new_mask[src_mask == LLayout.ceiling.value] = ALayout.ceil.value
        new_mask[src_mask == LLayout.left.value] = ALayout.wall1.value

        if LLayout.frontal.value in np.unique(src_mask):
            new_mask[src_mask == LLayout.frontal.value] = ALayout.wall2.value
            new_mask[src_mask == LLayout.right.value] = ALayout.wall3.value
        else:
            new_mask[src_mask == LLayout.right.value] = ALayout.wall2.value
        cv2.imwrite(str(dst_masks_dir / fpath.name), new_mask)


if __name__ == '__main__':
    app()
