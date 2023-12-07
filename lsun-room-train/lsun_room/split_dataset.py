import shutil
from pathlib import Path

import typer
from loguru import logger
from sklearn.model_selection import train_test_split
from tqdm import tqdm


app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def get_train_val_filenames(
        subdir: Path, val_size: float = 0.2
) -> tuple[list[str], list[str]]:
    fnames = []
    for fpath in subdir.glob('*'):
        fnames.append(fpath.stem)
    train_fnames, val_fnames = train_test_split(fnames, test_size=val_size)
    return train_fnames, val_fnames


def load_split(train_txt: Path, val_txt: Path) -> tuple[list[str], list[str]]:
    with open(train_txt) as f:
        train_fnames = f.read().strip().split('\n')
    with open(val_txt) as f:
        val_fnames = f.read().strip().split('\n')
    return train_fnames, val_fnames


@app.command()
def main(
        root_dir: Path,
        dst_dir: Path,
        train_txt: Path | None = None,
        val_txt: Path | None = None,
        val_size: float = 0.2,
        dst_exists_ok: bool = False,
):
    dst_dir.mkdir(parents=True, exist_ok=dst_exists_ok)
    if train_txt and val_txt:
        logger.info(f'Load train/val split')
        train_fnames, val_fnames = load_split(train_txt, val_txt)
    else:
        train_fnames, val_fnames = None, None
    for subdir in tqdm(list(root_dir.glob('*'))):
        if not subdir.is_dir():
            logger.info(f'{subdir} is not a directory. Skip.')
            continue
        if not train_fnames:
            # File names without extension
            train_fnames, val_fnames = get_train_val_filenames(
                subdir, val_size)
            with open(dst_dir / 'train.txt', 'w') as f:
                f.write('\n'.join(train_fnames))
            with open(dst_dir / 'val.txt', 'w') as f:
                f.write('\n'.join(val_fnames))
            logger.info(f'Created train/val split: {dst_dir / "train.txt"} '
                        f'(val.txt).')
        train_dir = dst_dir / 'train' / subdir.name
        train_dir.mkdir(parents=True, exist_ok=dst_exists_ok)
        val_dir = dst_dir / 'val' / subdir.name
        val_dir.mkdir(parents=True, exist_ok=dst_exists_ok)

        subdir_fpaths = list(subdir.glob('*'))
        for fpath in tqdm(subdir_fpaths, leave=False):
            if fpath.stem in train_fnames:
                shutil.copy(fpath, train_dir)
            elif fpath.stem in val_fnames:
                shutil.copy(fpath, val_dir)
            else:
                logger.warning(f'Unexpected file: {fpath}')


if __name__ == '__main__':
    app()
