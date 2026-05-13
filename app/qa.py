"""Q&A endpoint logic: page-text caching + DO Inference Router chat completions.

Design:
- The page content is sent by the extension only on the first question of a
  session. The backend caches it keyed by a stable page identifier (usually the
  page URL) so follow-up questions can be answered without re-sending the full
  page text from the client.
- The LLM still needs the page in its context on every call (LLMs are
  stateless), but the page appears once per request as a single system turn
  rather than being repeated inline with every user message.
- Prompt engineering keeps responses concise, grounded in the page, and free
  of filler.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import OrderedDict

import httpx
from fastapi import HTTPException

from app.config import get_settings
from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.preprocessing import preprocess_text
from app.schemas import QAMessage, QASource

import numpy as np

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a research assistant answering questions strictly about the "
    "document the user has open (academic paper, article, or docs page). "
    "The full document content is provided once below as PAGE CONTENT.\n\n"
    "Output a single JSON object with EXACTLY two keys and nothing else:\n"
    '  "answer": your reply text (plain text, no markdown)\n'
    '  "quote":  one verbatim excerpt from PAGE CONTENT that supports the '
    'answer, copied character-for-character. Use "" if no supporting quote '
    "exists.\n\n"
    'Example: {"answer": "Yes, the authors found X.", "quote": "We found '
    'that X holds across all three datasets."}\n\n'
    "Rules for the answer field:\n"
    "1. Answer only from PAGE CONTENT. If the answer is not present, reply "
    '"The page does not cover that." and set quote to "".\n'
    "2. Be concise. 1-2 sentences for most questions, up to 4 only for "
    "comparisons or enumerations.\n"
    "3. No preamble, no restating the question, no sign-offs, no filler "
    '("Great question", "Certainly", "Let me explain").\n'
    "4. Prefer specific details from the page (numbers, method names, author "
    "claims) over vague paraphrase.\n"
    '5. For yes/no questions, lead with "Yes" or "No" then one supporting '
    "clause.\n"
    "6. Do not hedge or speculate. Do not invent citations.\n\n"
    "Rules for the quote field:\n"
    "- The quote MUST appear in PAGE CONTENT exactly as written there. Do "
    "not paraphrase, summarize, or stitch distant passages together.\n"
    "- Aim for one full sentence (roughly 10-40 words).\n"
    "- Do not add ellipses, brackets, or any text not in the source."
)


class PageCache:
    """Tiny thread-unsafe LRU for preprocessed page text keyed by page id."""

    def __init__(self, max_size: int) -> None:
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._max_size = max(1, max_size)

    @staticmethod
    def _key(page_id: str) -> str:
        return hashlib.sha256(page_id.encode("utf-8")).hexdigest()

    def get(self, page_id: str) -> str | None:
        key = self._key(page_id)
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, page_id: str, page_text: str) -> None:
        key = self._key(page_id)
        self._cache[key] = page_text
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


_page_cache: PageCache | None = None


def get_page_cache() -> PageCache:
    global _page_cache
    if _page_cache is None:
        _page_cache = PageCache(max_size=get_settings().qa_page_cache_size)
    return _page_cache


def _resolve_page_text(
    page_id: str,
    page_text: str | None,
) -> tuple[str, bool]:
    """Return (text_to_use, was_cached). Updates cache when fresh text arrives."""
    cache = get_page_cache()
    cached = cache.get(page_id)

    if page_text:
        cleaned = preprocess_text(page_text)
        if not cleaned:
            raise HTTPException(
                status_code=400,
                detail="Page text is empty after preprocessing.",
            )
        cache.put(page_id, cleaned)
        return cleaned, cached is not None

    if cached is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "Page content not cached for this page. Resend the request "
                "with page_text included."
            ),
        )

    return cached, True


def _build_messages(
    page_text: str,
    history: list[QAMessage],
    question: str,
    max_page_chars: int,
    max_history_messages: int,
) -> list[dict[str, str]]:
    truncated_page = page_text[:max_page_chars]

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "system",
            "content": f"PAGE CONTENT:\n\n{truncated_page}",
        },
    ]

    # Keep only the most recent turns so the prompt stays bounded.
    bounded_history = history[-max_history_messages:] if max_history_messages > 0 else []
    for message in bounded_history:
        messages.append({"role": message.role, "content": message.content})

    messages.append({"role": "user", "content": question})
    return messages


def _call_chat_completions(messages: list[dict[str, str]]) -> str:
    settings = get_settings()
    if not settings.do_inference_token:
        raise HTTPException(
            status_code=500,
            detail="DO_INFERENCE_TOKEN is not configured on the server.",
        )

    payload = {
        "model": settings.chat_model,
        "messages": messages,
        "max_tokens": settings.qa_max_tokens,
        "temperature": settings.qa_temperature,
    }

    try:
        response = httpx.post(
            settings.do_chat_url,
            headers={"Authorization": f"Bearer {settings.do_inference_token}"},
            json=payload,
            timeout=settings.qa_timeout,
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Inference service timed out.")
    except httpx.HTTPError as exc:
        logger.exception("DO Inference chat transport error")
        raise HTTPException(status_code=502, detail=f"Inference transport error: {exc}")

    if response.status_code != 200:
        logger.error(
            "DO Inference chat error %s: %s",
            response.status_code,
            response.text,
        )
        raise HTTPException(
            status_code=502,
            detail=(
                f"Inference service error {response.status_code}: "
                f"{response.text[:200]}"
            ),
        )

    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        raise HTTPException(
            status_code=502,
            detail="Inference service returned no choices.",
        )

    answer = (choices[0].get("message") or {}).get("content", "").strip()
    if not answer:
        raise HTTPException(
            status_code=502,
            detail="Inference service returned an empty answer.",
        )
    return answer


def answer_question(
    question: str,
    page_id: str,
    page_text: str | None,
    history: list[QAMessage],
) -> tuple[str, bool, list[QASource]]:
    settings = get_settings()
    effective_text, was_cached = _resolve_page_text(page_id, page_text)
    messages = _build_messages(
        page_text=effective_text,
        history=history,
        question=question,
        max_page_chars=settings.qa_max_page_chars,
        max_history_messages=settings.qa_max_history_messages,
    )
    raw = _call_chat_completions(messages)
    answer_text, quote = _parse_answer_response(raw)

    sources: list[QASource] = []
    quote_source = _build_quote_source(quote, effective_text) if quote else None
    if quote_source is not None:
        sources.append(quote_source)

    refined = _find_sources(question, effective_text, top_k=3)
    seen = {_normalize_for_match(s.text) for s in sources}
    for src in refined:
        key = _normalize_for_match(src.text)
        if key in seen:
            continue
        sources.append(src)
        seen.add(key)
        if len(sources) >= 3:
            break

    return answer_text, was_cached, sources


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", flags=re.IGNORECASE | re.MULTILINE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", flags=re.DOTALL)


def _parse_answer_response(raw: str) -> tuple[str, str]:
    """Return (answer_text, quote). Falls back to (raw, "") on parse failure."""
    text = raw.strip()
    stripped = _JSON_FENCE_RE.sub("", text).strip()

    for candidate in (stripped, text):
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            answer = str(data.get("answer", "")).strip()
            quote = str(data.get("quote", "")).strip()
            if answer:
                return answer, quote

    match = _JSON_OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict):
                answer = str(data.get("answer", "")).strip()
                quote = str(data.get("quote", "")).strip()
                if answer:
                    return answer, quote
        except json.JSONDecodeError:
            pass

    return raw.strip(), ""


def _normalize_for_match(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _build_quote_source(quote: str, page_text: str) -> QASource | None:
    """Locate the LLM-emitted quote inside the page and return a tight source."""
    settings = get_settings()
    if not quote or len(quote.split()) < 4:
        return None

    norm_page = _normalize_for_match(page_text)
    norm_quote = _normalize_for_match(quote)
    if not norm_quote or norm_quote not in norm_page:
        # Relax: drop common trailing punctuation the LLM may have appended.
        trimmed = norm_quote.rstrip(".,;:!?\"')")
        if not trimmed or trimmed not in norm_page:
            return None

    prefix, suffix = extract_anchors(quote, settings.qa_anchor_words)
    return QASource(
        text=quote,
        score=1.0,
        chunk_id=-1,
        prefix=prefix,
        suffix=suffix,
    )


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“‘(])")


def _split_sentences(text: str, min_words: int = 5) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def extract_anchors(chunk_text_value: str, anchor_words: int) -> tuple[str, str]:
    """Return (prefix, suffix) — the first and last N words of the chunk.

    Used by the extension as DOM-search anchors: short enough to stay inside a
    single text node (so substring search succeeds), and used as the start/end
    of a Range so the highlight spans the full chunk even across element
    boundaries. When the chunk is shorter than N words, prefix == suffix.
    """
    words = chunk_text_value.split()
    if not words:
        return "", ""
    n = max(1, anchor_words)
    if len(words) <= n:
        joined = " ".join(words)
        return joined, joined
    return " ".join(words[:n]), " ".join(words[-n:])


def _find_sources(question: str, page_text: str, top_k: int = 3) -> list[QASource]:
    """Rank chunks by similarity, then refine each to its single best sentence.

    Two embedding passes:
      1. Question vs. every chunk to find the top-k coarse hits.
      2. Question vs. every sentence inside those top-k chunks to pick a tight
         anchor for highlighting.

    Falls back to the raw chunk text if a chunk can't be sentence-split.
    """
    settings = get_settings()
    try:
        chunks = chunk_text(
            page_text,
            max_words=settings.max_chunk_words,
            overlap_words=settings.overlap_words,
        )
        if not chunks:
            return []

        chunk_inputs = [question, *[c.text for c in chunks]]
        chunk_embeddings = embed_texts(chunk_inputs)
        question_emb = chunk_embeddings[0]
        chunk_scores = np.dot(chunk_embeddings[1:], question_emb)
        top_indices = [
            int(i) for i in np.argsort(chunk_scores)[::-1][:top_k]
            if chunk_scores[int(i)] > 0.3
        ]
        if not top_indices:
            return []

        sentence_pool: list[str] = []
        slices: list[tuple[int, int, int]] = []  # (chunk_idx, start, end)
        for chunk_idx in top_indices:
            sentences = _split_sentences(chunks[chunk_idx].text)
            if not sentences:
                slices.append((chunk_idx, -1, -1))
                continue
            start = len(sentence_pool)
            sentence_pool.extend(sentences)
            slices.append((chunk_idx, start, len(sentence_pool)))

        sentence_scores = (
            np.dot(embed_texts(sentence_pool), question_emb)
            if sentence_pool
            else np.empty(0)
        )

        neighbors = max(0, settings.qa_passage_neighbors)
        sources: list[QASource] = []
        for chunk_idx, start, end in slices:
            chunk = chunks[chunk_idx]
            chunk_score = float(chunk_scores[chunk_idx])
            if start < 0:
                text_value = chunk.text
            else:
                chunk_sentences = sentence_pool[start:end]
                local = sentence_scores[start:end]
                best = int(np.argmax(local))
                win_start = max(0, best - neighbors)
                win_end = min(len(chunk_sentences), best + neighbors + 1)
                text_value = " ".join(chunk_sentences[win_start:win_end])
            prefix, suffix = extract_anchors(text_value, settings.qa_anchor_words)
            sources.append(
                QASource(
                    text=text_value,
                    score=round(chunk_score, 4),
                    chunk_id=chunk.chunk_id,
                    prefix=prefix,
                    suffix=suffix,
                )
            )
        return sources
    except Exception:
        logger.exception("Failed to find sources")
        return []
