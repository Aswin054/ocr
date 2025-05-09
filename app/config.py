import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/license_plate_detector.pt")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))  # Render uses 8000 by default
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    WORKERS: int = int(os.getenv("WORKERS", "1"))
    
    class Config:
        case_sensitive = True

settings = Settings()