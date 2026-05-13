from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RankChunksRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    page_text: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1)
    url: str | None = None

    @field_validator("prompt", "page_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class RankedChunk(BaseModel):
    rank: int
    score: float
    text: str
    explanation: str
    prefix: str = ""
    suffix: str = ""


class RankChunksResponse(BaseModel):
    chunks: list[RankedChunk]


class QAMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def strip_content(cls, value: str) -> str:
        return value.strip()


class QARequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    page_id: str = Field(..., min_length=1, max_length=2048)
    page_text: str | None = None
    history: list[QAMessage] = Field(default_factory=list)

    @field_validator("question", "page_id")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()

    @field_validator("page_text")
    @classmethod
    def strip_page_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class QASource(BaseModel):
    text: str
    score: float
    chunk_id: int
    prefix: str
    suffix: str


class QAResponse(BaseModel):
    answer: str
    cached: bool
    sources: list[QASource] = Field(default_factory=list)
