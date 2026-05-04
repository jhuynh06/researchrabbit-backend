from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    do_inference_token: str = ""
    do_inference_url: str = "https://inference.do-ai.run/v1/embeddings"
    embedding_model: str = "qwen3-embedding-0.6b"
    embedding_timeout: float = 10.0
    max_chunk_words: int = 850
    overlap_words: int = 125
    candidate_explanation: str = "Highest semantic match to the prompt."
    min_page_text_chars: int = 100
    default_top_k: int = 5
    max_top_k: int = 10
    allowed_origins: str = "chrome-extension://*"


@lru_cache
def get_settings() -> Settings:
    return Settings()
