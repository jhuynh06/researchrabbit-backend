from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.qa import answer_question
from app.ranking import rank_chunks
from app.schemas import (
    QARequest,
    QAResponse,
    RankChunksRequest,
    RankChunksResponse,
)

app = FastAPI(title="ResearchRabbit Backend")

settings = get_settings()
origins = [o.strip() for o in settings.allowed_origins.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"chrome-extension://.*",
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/rank-chunks", response_model=RankChunksResponse)
def rank_chunks_endpoint(request: RankChunksRequest) -> RankChunksResponse:
    if len(request.page_text) < settings.min_page_text_chars:
        raise HTTPException(
            status_code=400,
            detail="Extracted page text is too short. Open an HTML research page and try again.",
        )

    chunks = rank_chunks(
        prompt=request.prompt,
        page_text=request.page_text,
        top_k=request.top_k,
    )
    return RankChunksResponse(chunks=chunks)


@app.post("/qa", response_model=QAResponse)
def qa_endpoint(request: QARequest) -> QAResponse:
    if request.page_text is not None and len(request.page_text) < settings.min_page_text_chars:
        raise HTTPException(
            status_code=400,
            detail="Extracted page text is too short. Open an HTML research page and try again.",
        )

    answer, cached, sources = answer_question(
        question=request.question,
        page_id=request.page_id,
        page_text=request.page_text,
        history=request.history,
    )
    return QAResponse(answer=answer, cached=cached, sources=sources)
