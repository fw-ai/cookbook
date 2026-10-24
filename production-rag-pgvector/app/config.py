from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    fireworks_api_key: str
    fireworks_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"
    fireworks_chat_model: str = "accounts/fireworks/models/llama-v3p1-70b-instruct"
    database_url: str = "postgresql://rag:rag@localhost:5432/rag"
    embedding_dim: int = 768
    retrieval_top_k: int = 5


settings = Settings()
