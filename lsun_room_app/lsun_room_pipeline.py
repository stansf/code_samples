"""Classes to work with room layout estimation using lsun-room model."""
import io
import os
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime
from PIL import Image

from .utils import blend_np

WEIGHTS_DEFAULT_PATH = Path(__file__).parent / 'lsun_room.onnx'


class DataLoader:
    def process(self: 'DataLoader', im_bytes: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(im_bytes)).convert('RGB')
        return img


class Preprocessor:
    """Preprocess input image."""

    def __init__(self: 'Preprocessor') -> None:
        self.mean = 255 * 0.5
        self.std = 255 * 0.5
        self.img_size = (320, 320)

    def process(self: 'Preprocessor', image: Image.Image) -> np.array:
        src_size = image.size
        image = np.array(image)
        resized = cv2.resize(image, self.img_size, cv2.INTER_LINEAR).astype(
            np.float32)
        resized = np.transpose(resized, (2, 0, 1))[None, ...]
        preprocessed = (resized - self.mean) / self.std
        return preprocessed, src_size


class LsunRoomEstimator:
    """Class that computes room layout."""

    def __init__(
            self: 'LsunRoomEstimator', weights_path: Optional[Path] = None
    ) -> None:
        weights_path = weights_path or WEIGHTS_DEFAULT_PATH
        self.ort_session = self._get_model(weights_path)
        self.preprocessor = Preprocessor()

        # Fake run to init TRT model optimization
        self._process_sample(
            Image.fromarray(
                np.random.randint(0, 256, size=(320, 320, 3)).astype(np.uint8)
            )
        )

    def process_list(
            self: 'LsunRoomEstimator',
            data: List[Tuple[Image.Image, bool]],
    ) -> List[np.ndarray]:
        return [self._process_sample(*t) for t in data]

    def _get_model(
            self: 'LsunRoomEstimator', weights_path: Path
    ) -> onnxruntime.InferenceSession:
        """Load model with pretrained weights."""
        # providers = ['CUDAExecutionProvider']
        providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
        ort_session = onnxruntime.InferenceSession(
            str(weights_path),
            providers=providers
        )
        return ort_session

    def _process_sample(
            self: 'LsunRoomEstimator',
            sample: Image.Image,
            with_blend: bool = False
    ) -> np.ndarray:
        preprocessed, src_size = self.preprocessor.process(sample)
        _, result = self.ort_session.run(
            ['onnx::ArgMax_994', '995'],
            {'input.1': preprocessed}
        )
        result = result.squeeze(0).astype(np.uint8)
        if with_blend:
            result = blend_np(preprocessed, result)
        room_layout_mask = cv2.resize(result, src_size,
                                      interpolation=cv2.INTER_NEAREST_EXACT)
        room_layout_mask = cv2.cvtColor(room_layout_mask, cv2.COLOR_BGR2RGB)
        return room_layout_mask


class PostProcess:
    """Post-process results of room layout estimation."""

    def __init__(self: 'PostProcess', cmap: str = 'gray_r') -> None:
        self.cmap = cmap

    def process(self: 'PostProcess', mask_array: np.ndarray) -> bytes:
        """Encode image as bytes."""
        success, encoded_image = cv2.imencode('.png', mask_array)
        bytes_image = encoded_image.tobytes()
        assert success
        return bytes_image


def main() -> None:
    import dotenv

    dotenv.load_dotenv()

    images_dir = os.getenv('IMAGES_DIR')
    out_dir = os.getenv('OUT_DIR')
    assert images_dir is not None
    assert out_dir is not None
    images_dir = Path(images_dir)
    out_dir = Path(out_dir)

    dl = DataLoader()
    de = LsunRoomEstimator()
    dpp = PostProcess()

    with open(images_dir / 'img1.jpg', 'rb') as f:
        b_img = f.read()
    img = dl.process(b_img)
    mask = de.process_list([(img, True)])
    print(mask[0].mean())
    out = dpp.process(mask[0])
    with open(out_dir / 'out1.png', 'wb') as f:
        f.write(out)


if __name__ == '__main__':
    main()

