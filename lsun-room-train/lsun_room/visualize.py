from pathlib import Path

import cv2
import numpy as np
import typer
from loguru import logger

from datasets.custom import (CustomDataset, get_colorized_layout,
                              denormalize_image)


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
        dataset_dir: Path,
        phase: str = 'training',
        batch_size: int = 4,
        as_loader: bool = False
):
    dataset = ArtixelDataset(phase, dataset_dir, (320, 320))
    if as_loader:
        print('Not implemented.')
        return 1
    logger.info(f'Dataset size: {len(dataset)}')
    for i, item in enumerate(dataset):
        logger.info(f'Read {i}-th sample')
        image = item['image']
        mask = item['label']
        edge = item['edge']

        color_mask = get_colorized_layout(mask.numpy())

        edge = (edge * 255).numpy().astype(np.uint8)
        edge = np.tile(np.expand_dims(edge, 2), (1, 1, 3))

        denorm_image = denormalize_image(image, dataset.norm_mean,
                                         dataset.norm_std).numpy() * 255
        denorm_image = np.clip(denorm_image, 0, 255).astype(np.uint8)
        denorm_image = np.transpose(denorm_image, (1, 2, 0))
        denorm_image = cv2.cvtColor(denorm_image, cv2.COLOR_RGB2BGR)

        vis_mask = cv2.addWeighted(denorm_image, 0.8, color_mask, 0.3, 1.0)
        vis_mask = cv2.addWeighted(vis_mask, 0.8, edge, 0.3, 1.0)
        vis = np.hstack([denorm_image, color_mask, edge, vis_mask])
        cv2.imshow('Visualize dataset', vis)
        k = cv2.waitKey()
        if k == 27:  # ESC
            return


if __name__ == '__main__':
    app()
