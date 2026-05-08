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

    # Chat / Q&A settings (DO Inference Router is OpenAI-compatible).
    do_chat_url: str = "https://inference.do-ai.run/v1/chat/completions"
    chat_model: str = "anthropic-claude-haiku-4.5"
    qa_timeout: float = 45.0
    qa_max_tokens: int = 320
    qa_temperature: float = 0.2
    qa_max_page_chars: int = 60000
    qa_max_history_messages: int = 12
    qa_page_cache_size: int = 128


@lru_cache
def get_settings() -> Settings:
    return Settings()
