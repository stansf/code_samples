from functools import lru_cache

import typer
from config import get_settings
from errors import DatabaseCredentialsError
from loguru import logger
from models import AIStorageImage  # noqa
from sqlalchemy.exc import OperationalError
from sqlalchemy.future import Engine
from sqlmodel import SQLModel, create_engine

app = typer.Typer(add_completion=False, pretty_exceptions_short=True,
                  pretty_exceptions_enable=False)


# DB_USERNAME = os.getenv('DB_USERNAME')
# DB_PASSWORD = os.getenv('DB_PASSWORD')
# DB_URL = os.getenv('DB_URL')
# DB_PORT = os.getenv('DB_PORT', '5432')
# DB_NAME = os.getenv('DB_NAME')


@lru_cache()
def get_engine() -> Engine:
    # if not (DB_USERNAME and DB_PASSWORD and DB_URL and DB_NAME):
    #     d = {'DB_USERNAME': DB_USERNAME, 'DB_PASSWORD': DB_PASSWORD,
    #          'DB_URL': DB_URL, 'DB_PORT': DB_PORT, 'DB_NAME': DB_NAME,}
    #     raise DatabaseCredentialsError(
    #         f'Not enough database credentials\n{pformat(d)}')
    # pg_url = (f'postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_URL}:{DB_PORT}/'
    #           f'{DB_NAME}')
    engine = create_engine(get_settings().pg_dns)
    logger.info(f'Connected to {get_settings().pg_dns}')
    return engine


@app.command('create_table')
def create_db_and_table() -> None:
    confirm = input('Do you really want to create a table? [y/N]')
    if confirm.lower() == 'y':
        logger.info('Creating table...')
        try:
            engine = get_engine()
            print(engine)
            SQLModel.metadata.create_all(engine)
            logger.info('Complete.')
        except OperationalError as e:
            logger.error(f'Failed. Can not connect to server. Reason: {e}')
        except DatabaseCredentialsError as e:
            logger.error(f'Failed. Engine error. Reason: {e}')

    else:
        logger.info('Stop. Table will not be created.')


if __name__ == '__main__':
    app()
