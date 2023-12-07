from enum import Enum
from typing import Optional

from sqlmodel import Field, SQLModel


class ImageType(str, Enum):
    input: int = 'input'  # noqa
    output: int = 'output'
    crop: int = 'crop'
    crop_rgba: int = 'crop_rgba'


class AIStorageImage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)  # noqa
    image_type: ImageType
    gcs_blob_name: str
    user_id: Optional[int] = Field(default=None)
