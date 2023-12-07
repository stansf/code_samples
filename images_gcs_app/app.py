from http import HTTPStatus
from pprint import pformat
from typing import Annotated, Optional

import shortuuid
from db_utils import get_engine
from fastapi import FastAPI, File, Header, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from gcp_utils import create_blob_dir_name, get_gcs_routine
from loguru import logger
from models import AIStorageImage, ImageType
from sqlmodel import Session

from version import __version__

app = FastAPI(
    title='User images API',
    description='API to interact with images originally stored on GCS'
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.post('/api/v1/upload')
async def upload_image(
        image_type: ImageType,
        file: bytes = File(...),
        image_name: Optional[str] = None,
        user_id: Optional[int] = None,
        session_id: Annotated[Optional[str], Header()] = None,
) -> int:
    """Upload an image on GCS and save record in database. Return image id."""
    blob_dir = create_blob_dir_name(session_id)
    if image_name is None:
        image_name = f'{shortuuid.uuid()}.webp'
    blob_name = f'{blob_dir}/{image_name}'
    success, reason = get_gcs_routine().upload_bytes(file, blob_name)
    if not success:
        logger.warning(
            f'Something goes wrong in uploading. Reason: {reason}')
        raise HTTPException(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            detail={'status': 'Can not load image to GCS.',
                    'reason': reason}
        )

    storage_image = AIStorageImage(
        image_type=image_type, gcs_blob_name=blob_name, user_id=user_id)

    engine = get_engine()
    with Session(engine) as session:
        session.add(storage_image)
        session.commit()
        session.refresh(storage_image)
        return storage_image.id


@app.get('/api/v1/images/url/{image_id}')
def get_image_signed_url(
        image_id: int
) -> str:
    """Get SignedURL for given image id."""
    engine = get_engine()
    with Session(engine) as session:
        logger.debug(f'Get entity with image_id {image_id}')
        image_entity = session.get(AIStorageImage, image_id)
        logger.debug(f'Got:\n{pformat(image_entity)}')
    if image_entity is None:
        raise HTTPException(HTTPStatus.NOT_FOUND,
                            detail=f'Not found blob for image {image_id}')
    url = get_gcs_routine().get_signed_url(image_entity.gcs_blob_name)
    return url


@app.get(
    '/api/v1/images/{image_id}',
    responses={
        HTTPStatus.OK: {
            'content': {'image/webp': {}}
        }
    },
    response_class=Response
)
def get_image_bytes(
        image_id: int
) -> Response:
    """Get image itself (as bytes) for given image id."""
    engine = get_engine()
    with Session(engine) as session:
        logger.debug(f'Get entity with image_id {image_id}')
        image_entity = session.get(AIStorageImage, image_id)
        logger.debug(f'Got:\n{pformat(image_entity)}')
    if image_entity is None:
        raise HTTPException(HTTPStatus.NOT_FOUND,
                            detail=f'Not found blob for image {image_id}')
    result = get_gcs_routine().download(image_entity.gcs_blob_name)
    if result is None:
        raise HTTPException(HTTPStatus.INTERNAL_SERVER_ERROR,
                            detail=f'Can not download image {image_id}')
    b, media_type = result
    return Response(b, media_type=media_type)


@app.get('/api/v1/version')
def version() -> str:
    return __version__
