import datetime as dt
import io
import os
from functools import lru_cache
from typing import Optional, Tuple, Union
from uuid import uuid4

import cv2
import numpy as np
import pytz
from google.cloud import storage
from google.cloud.storage import Blob, Bucket
from loguru import logger
from PIL import Image

EXT_TO_CONTENT_TYPE = {
    '.jpg': 'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.png': 'image/png',
    '.webp': 'image/webp'
}


class GCSRoutine:
    """Routines for uploading to GCS bucket."""

    def __init__(
            self: 'GCSRoutine',
            bucket_name: str,
            url_expire_timeout: int = 15
    ) -> None:
        self._bucket_name = bucket_name
        client = storage.Client()
        self._bucket: Bucket = client.bucket(self._bucket_name)
        self._url_expire_timeout = url_expire_timeout

    def check_blob_exist(self: 'GCSRoutine', blob: Union[str, Blob]) -> bool:
        if isinstance(blob, str):
            blob = self._bucket.blob(blob)
        return blob.exists()

    def _upload(
            self: 'GCSRoutine',
            b: io.BytesIO,
            blob_name: str,
            file_ext: str
    ) -> Tuple[bool, Optional[str]]:
        try:
            content_type = EXT_TO_CONTENT_TYPE.get(file_ext)
            blob: Blob = self._bucket.blob(blob_name)
            if self.check_blob_exist(blob_name):
                return False, f'Blob with name {blob_name} already exists.'
            blob.upload_from_string(b.getvalue(), content_type=content_type)
            return True, None
        except Exception as e:
            logger.error(f'Can not upload file: {e}')
            return False, str(e)

    def _parse_blob_name(
            self: 'GCSRoutine', blob_name: str
    ) -> Tuple[str, str]:
        name, file_ext = os.path.splitext(blob_name)
        if file_ext not in ('.png', '.jpg', '.jpeg', '.webp'):
            raise RuntimeError(f'Unexpected file format: {file_ext}')
        return name, file_ext

    def get_signed_url(self: 'GCSRoutine', blob_name: str) -> Optional[str]:
        """Create a signed url for the given blob."""
        blob = self._bucket.blob(blob_name)
        if not self.check_blob_exist(blob_name):
            logger.warning(f'Not found blob with name {blob_name}')
            return None
        url = blob.generate_signed_url(
            version='v4',
            expiration=dt.timedelta(minutes=self._url_expire_timeout),
            method='GET',
        )
        return url

    def download(
            self: 'GCSRoutine', blob_name: str
    ) -> Optional[Tuple[bytes, str]]:
        """Download an image as bytes for the given blob."""
        blob = self._bucket.blob(blob_name)
        if not self.check_blob_exist(blob_name):
            logger.warning(f'Not found blob with name {blob_name}')
            return None
        _, ext = os.path.splitext(blob_name)
        media_type = EXT_TO_CONTENT_TYPE.get(ext)
        if media_type is None:
            logger.error(f'Unknown image format {ext} ({blob_name})')
        b = blob.download_as_bytes()
        return b, media_type

    def upload_bytes(
            self: 'GCSRoutine', image: bytes, blob_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Upload bytes to GCS bucket to the blob with name blob_name."""
        name, file_ext = self._parse_blob_name(blob_name)
        b = io.BytesIO(image)
        success, reason = self._upload(b, blob_name, file_ext)
        return success, reason

    def upload_pil(
            self: 'GCSRoutine', image: Image.Image, blob_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Upload PIL image to GCS bucket to the blob with name blob_name."""
        name, file_ext = self._parse_blob_name(blob_name)
        b = io.BytesIO()
        if file_ext == '.jpg':
            file_ext = '.jpeg'
        pil_format = file_ext[1:]
        image.save(b, pil_format)
        success, reason = self._upload(b, blob_name, file_ext)
        return success, reason

    def upload_cv2(
            self: 'GCSRoutine', image: np.ndarray, blob_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Upload OpenCV image to GCS bucket to the blob with name blob_name.
        """  # noqa
        name, file_ext = self._parse_blob_name(blob_name)
        if file_ext == '.jpeg':
            file_ext = '.jpg'
        b = io.BytesIO(cv2.imencode(file_ext, image)[1])
        success, reason = self._upload(b, blob_name, file_ext)
        return success, reason


@lru_cache
def get_gcs_routine() -> GCSRoutine:
    gcs_bucket_name = os.getenv('GCS_BUCKET')
    if gcs_bucket_name is None:
        raise RuntimeError('Not found environment variable GCS_BUCKET.')
    return GCSRoutine(gcs_bucket_name)


def create_blob_dir_name(session_id: Optional[str] = None) -> str:
    """Create a unique 'directory' name for bucket."""
    unique_suffix = uuid4().hex
    session_id = session_id or unique_suffix
    tz = pytz.timezone('Europe/Moscow')
    prefix = dt.datetime.now(tz).strftime('%y-%m-%d')
    return f'{prefix}_{session_id}'


# if __name__ == '__main__':
#     nname = '23-10-13_2317cac2d48a482185d3118d98469cc5/tmp2.webp'
#     r = get_gcs_uploader()
#     print(r._check_blob_exist(nname))
    # img = Image.open('image.webp')
    # with open('flat.jpeg', 'rb') as f:
    #     b_img = f.read()
    # success = r.upload_bytes(b_img, nname)
    # print(success)
    # blob = r.upload_pil(img, f'{create_blob_dir_name()}/tmp.webp')
    # print(blob)
