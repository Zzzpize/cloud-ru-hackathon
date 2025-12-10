from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    db_path: str = "checklists_db"

    class Config:
        env_file = ".env"


settings = Settings()