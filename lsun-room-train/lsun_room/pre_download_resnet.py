from .trainer.core import LayoutSeg
from loguru import logger


def main():
    model = LayoutSeg()
    logger.info('Model created')


if __name__ == '__main__':
    main()
