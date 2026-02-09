import os
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application Info
    APP_NAME: str = "Sales Intelligence Hub"
    APP_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = False
    
    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str = "sales_intelligence"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    
    # Backend API (for Dashboard)
    API_SERVER: str = "localhost"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # ML Service Params
    FORECAST_HORIZON_DAYS: int = 30
    LEAD_SCORE_THRESHOLD: float = 0.5
    
    # External APIs
    OPENAI_API_KEY: str
    
    # Paths
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    LOGS_DIR: str = os.path.join(BASE_DIR, "logs")

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()

# Create directories if they don't exist
settings = get_settings()
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)
os.makedirs(settings.LOGS_DIR, exist_ok=True)
