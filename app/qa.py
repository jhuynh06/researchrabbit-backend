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
import logging
from collections import OrderedDict

import httpx
from fastapi import HTTPException

from app.config import get_settings
from app.preprocessing import preprocess_text
from app.schemas import QAMessage

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a research assistant answering questions strictly about the "
    "document the user has open (academic paper, article, or docs page). "
    "The full document content is provided once below as PAGE CONTENT.\n\n"
    "Rules for every reply:\n"
    "1. Answer only from PAGE CONTENT. If the answer is not present, reply "
    '"The page does not cover that." and stop.\n'
    "2. Be concise. 1-2 sentences for most questions, up to 4 only for "
    "comparisons or enumerations.\n"
    "3. No preamble, no restating the question, no sign-offs, no filler "
    '("Great question", "Certainly", "Let me explain").\n'
    "4. Prefer specific details from the page (numbers, method names, author "
    "claims) over vague paraphrase.\n"
    '5. For yes/no questions, lead with "Yes" or "No" then one supporting '
    "clause.\n"
    "6. Do not hedge or speculate. Do not invent citations.\n"
    "7. Plain text only. Use a short bulleted list only when the user "
    "explicitly asks for a list of 3+ items."
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
) -> tuple[str, bool]:
    settings = get_settings()
    effective_text, was_cached = _resolve_page_text(page_id, page_text)
    messages = _build_messages(
        page_text=effective_text,
        history=history,
        question=question,
        max_page_chars=settings.qa_max_page_chars,
        max_history_messages=settings.qa_max_history_messages,
    )
    answer = _call_chat_completions(messages)
    return answer, was_cached
