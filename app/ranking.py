import numpy as np

from app.cache import cache
from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import embed_texts
from app.preprocessing import preprocess_text
from app.schemas import RankedChunk


def cosine_scores(prompt_embedding: np.ndarray, chunk_embeddings: np.ndarray) -> np.ndarray:
    return np.dot(chunk_embeddings, prompt_embedding)


def rank_chunks(prompt: str, page_text: str, top_k: int | None = None) -> list[RankedChunk]:
    settings = get_settings()
    clean_text = preprocess_text(page_text)
    chunks = chunk_text(
        clean_text,
        max_words=settings.max_chunk_words,
        overlap_words=settings.overlap_words,
    )

    if not chunks:
        return []

    requested_top_k = top_k or settings.default_top_k
    effective_top_k = min(requested_top_k, settings.max_top_k, len(chunks))

    # Check cache for chunk embeddings
    cached_chunk_embeddings = cache.get(clean_text)

    if cached_chunk_embeddings is not None and cached_chunk_embeddings.shape[0] == len(chunks):
        # Only embed the prompt
        prompt_embedding = embed_texts([prompt])[0]
        chunk_embeddings = cached_chunk_embeddings
    else:
        # Embed everything and cache the chunk embeddings
        texts = [prompt, *[chunk.text for chunk in chunks]]
        embeddings = embed_texts(texts)
        prompt_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]
        cache.put(clean_text, chunk_embeddings)

    scores = cosine_scores(prompt_embedding, chunk_embeddings)
    ranked_indexes = np.argsort(scores)[::-1][:effective_top_k]

    return [
        RankedChunk(
            rank=rank,
            score=round(float(scores[index]), 4),
            text=chunks[index].text,
            explanation=settings.candidate_explanation,
        )
        for rank, index in enumerate(ranked_indexes, start=1)
    ]
