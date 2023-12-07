from functools import lru_cache
from pprint import pprint

from pydantic import BaseSettings, PostgresDsn


class Settings(BaseSettings):
    """Base settings for ImagesGCSApp."""

    pg_dns: PostgresDsn  # e.g. 'postgresql://user:pass@localhost:5432/foobar'

    class Config:
        """Config of settings class."""

        env_file = '.env'
        env_prefix = 'images_gcs_api_'


@lru_cache()
def get_settings() -> Settings:
    return Settings()


if __name__ == '__main__':
    pprint(get_settings().dict())
    print(type(get_settings().pg_dns))
    print(get_settings().pg_dns)
    print(str(get_settings().pg_dns))
