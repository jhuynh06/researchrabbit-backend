import numpy as np

from app.cache import cache
from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import embed_texts
from app.preprocessing import preprocess_text
from app.qa import _split_sentences, extract_anchors
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

    cached_chunk_embeddings = cache.get(clean_text)

    if cached_chunk_embeddings is not None and cached_chunk_embeddings.shape[0] == len(chunks):
        prompt_embedding = embed_texts([prompt])[0]
        chunk_embeddings = cached_chunk_embeddings
    else:
        texts = [prompt, *[chunk.text for chunk in chunks]]
        embeddings = embed_texts(texts)
        prompt_embedding = embeddings[0]
        chunk_embeddings = embeddings[1:]
        cache.put(clean_text, chunk_embeddings)

    scores = cosine_scores(prompt_embedding, chunk_embeddings)
    ranked_indexes = [int(i) for i in np.argsort(scores)[::-1][:effective_top_k]]

    # Sentence-level refinement: pick the single best sentence per top chunk so
    # the search-panel snippet (and the resulting highlight) is tight instead
    # of paragraph-sized.
    sentence_pool: list[str] = []
    slices: list[tuple[int, int]] = []  # (start, end) per ranked index
    for chunk_idx in ranked_indexes:
        sentences = _split_sentences(chunks[chunk_idx].text)
        if not sentences:
            slices.append((-1, -1))
            continue
        start = len(sentence_pool)
        sentence_pool.extend(sentences)
        slices.append((start, len(sentence_pool)))

    sentence_scores = (
        np.dot(embed_texts(sentence_pool), prompt_embedding)
        if sentence_pool
        else np.empty(0)
    )

    neighbors = max(0, getattr(settings, "qa_passage_neighbors", 2))
    results: list[RankedChunk] = []
    for rank_pos, (chunk_idx, (sent_start, sent_end)) in enumerate(
        zip(ranked_indexes, slices), start=1
    ):
        if sent_start < 0:
            text_value = chunks[chunk_idx].text
        else:
            chunk_sentences = sentence_pool[sent_start:sent_end]
            local = sentence_scores[sent_start:sent_end]
            best = int(np.argmax(local))
            win_start = max(0, best - neighbors)
            win_end = min(len(chunk_sentences), best + neighbors + 1)
            text_value = " ".join(chunk_sentences[win_start:win_end])
        prefix, suffix = extract_anchors(text_value, settings.qa_anchor_words)
        results.append(
            RankedChunk(
                rank=rank_pos,
                score=round(float(scores[chunk_idx]), 4),
                text=text_value,
                explanation=settings.candidate_explanation,
                prefix=prefix,
                suffix=suffix,
            )
        )
    return results
