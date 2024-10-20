from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str
    APP_VERSION: str
    
    MONGODB_URL: str
    MONGODB_DATABASE: str
    
    DATA_CLASSIC_PATH: str
    MODEL_CLASSIC_PATH: str
    DATASET_TRAIN_NAME: str
    
    COHERE_API_KEY: str
    MODEL_ID: str
    TEMPERATURE: float

    class Config:
        env_file = ".env"


def get_settings():
    return Settings()