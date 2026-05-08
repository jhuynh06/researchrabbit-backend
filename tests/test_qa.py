from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app import qa
from app.qa import PageCache, answer_question
from app.schemas import QAMessage


SETTINGS = SimpleNamespace(
    do_inference_token="test-token",
    do_chat_url="https://inference.do-ai.run/v1/chat/completions",
    chat_model="alibaba-qwen3-32b",
    qa_timeout=10.0,
    qa_max_tokens=200,
    qa_temperature=0.2,
    qa_max_page_chars=60000,
    qa_max_history_messages=12,
    qa_page_cache_size=16,
    max_chunk_words=850,
    overlap_words=125,
)


@pytest.fixture(autouse=True)
def _reset_page_cache(monkeypatch):
    """Give every test a clean cache and a predictable settings object."""
    fresh_cache = PageCache(max_size=16)
    monkeypatch.setattr(qa, "_page_cache", fresh_cache)
    monkeypatch.setattr(qa, "get_page_cache", lambda: fresh_cache)
    monkeypatch.setattr(qa, "get_settings", lambda: SETTINGS)
    yield


def _mock_chat_response(content: str = "Short answer.", status_code: int = 200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": content}}]
    }
    response.text = content
    return response


def test_page_cache_lru_eviction():
    cache = PageCache(max_size=2)
    cache.put("a", "alpha")
    cache.put("b", "beta")
    cache.put("c", "gamma")  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == "beta"
    assert cache.get("c") == "gamma"


def test_page_cache_get_updates_recency():
    cache = PageCache(max_size=2)
    cache.put("a", "alpha")
    cache.put("b", "beta")
    # Touch "a" so "b" becomes least-recent.
    assert cache.get("a") == "alpha"
    cache.put("c", "gamma")
    assert cache.get("b") is None
    assert cache.get("a") == "alpha"


@patch("app.qa.httpx.post")
@patch("app.qa._find_sources", return_value=[])
def test_answer_question_caches_page_text_on_first_call(_mock_sources, mock_post):
    mock_post.return_value = _mock_chat_response("42 samples.")

    answer, cached, sources = answer_question(
        question="How many samples?",
        page_id="https://example.com/paper",
        page_text="The study uses 42 samples from the benchmark dataset." * 5,
        history=[],
    )

    assert answer == "42 samples."
    assert cached is False
    # Page should now be cached for this page_id.
    assert qa.get_page_cache().get("https://example.com/paper") is not None

    # Verify payload sent to DO.
    call_kwargs = mock_post.call_args.kwargs
    sent = call_kwargs["json"]
    assert sent["model"] == "alibaba-qwen3-32b"
    assert sent["temperature"] == 0.2
    # System prompt + page-content system turn + user question.
    roles = [m["role"] for m in sent["messages"]]
    assert roles[0] == "system"
    assert roles[1] == "system"
    assert roles[-1] == "user"
    assert "PAGE CONTENT" in sent["messages"][1]["content"]
    assert sent["messages"][-1]["content"] == "How many samples?"


@patch("app.qa.httpx.post")
@patch("app.qa._find_sources", return_value=[])
def test_answer_question_reuses_cached_page_text(_mock_sources, mock_post):
    mock_post.return_value = _mock_chat_response("First answer.")
    answer_question(
        question="Q1",
        page_id="https://example.com/paper",
        page_text="Long page body with dataset and results." * 10,
        history=[],
    )

    mock_post.return_value = _mock_chat_response("Second answer.")
    answer, cached, sources = answer_question(
        question="Q2",
        page_id="https://example.com/paper",
        page_text=None,
        history=[QAMessage(role="user", content="Q1"), QAMessage(role="assistant", content="First answer.")],
    )

    assert answer == "Second answer."
    assert cached is True

    sent = mock_post.call_args.kwargs["json"]
    # History should be included after the page-content system turn.
    roles = [m["role"] for m in sent["messages"]]
    assert roles == ["system", "system", "user", "assistant", "user"]
    assert sent["messages"][-1]["content"] == "Q2"


def test_answer_question_missing_page_without_cache_raises_409():
    with pytest.raises(HTTPException) as exc_info:
        answer_question(
            question="What is this?",
            page_id="https://example.com/new",
            page_text=None,
            history=[],
        )
    assert exc_info.value.status_code == 409


@patch("app.qa.httpx.post", side_effect=httpx.TimeoutException("timeout"))
def test_answer_question_handles_timeout(_mock_post):
    with pytest.raises(HTTPException) as exc_info:
        answer_question(
            question="Q",
            page_id="https://example.com/x",
            page_text="Plenty of page text here to process." * 10,
            history=[],
        )
    assert exc_info.value.status_code == 504


@patch("app.qa.httpx.post")
def test_answer_question_propagates_upstream_errors(mock_post):
    mock_post.return_value = _mock_chat_response("", status_code=500)

    with pytest.raises(HTTPException) as exc_info:
        answer_question(
            question="Q",
            page_id="https://example.com/x",
            page_text="Plenty of page text here to process." * 10,
            history=[],
        )
    assert exc_info.value.status_code == 502


@patch("app.qa.httpx.post")
def test_answer_question_empty_response_raises(mock_post):
    mock_post.return_value = _mock_chat_response("")
    with pytest.raises(HTTPException) as exc_info:
        answer_question(
            question="Q",
            page_id="https://example.com/x",
            page_text="Plenty of page text here to process." * 10,
            history=[],
        )
    assert exc_info.value.status_code == 502


@patch("app.qa.httpx.post")
@patch("app.qa._find_sources", return_value=[])
def test_answer_question_truncates_long_page(_mock_sources, mock_post):
    mock_post.return_value = _mock_chat_response("ok")
    long_page = "A" * (SETTINGS.qa_max_page_chars + 5000)
    answer_question(
        question="Q",
        page_id="pid",
        page_text=long_page,
        history=[],
    )
    sent = mock_post.call_args.kwargs["json"]
    page_message = sent["messages"][1]["content"]
    # "PAGE CONTENT:\n\n" prefix + truncated body.
    assert len(page_message) <= SETTINGS.qa_max_page_chars + len("PAGE CONTENT:\n\n")


@patch("app.qa.httpx.post")
@patch("app.qa._find_sources", return_value=[])
def test_answer_question_bounds_history(_mock_sources, mock_post):
    mock_post.return_value = _mock_chat_response("ok")
    history = [
        QAMessage(role="user" if i % 2 == 0 else "assistant", content=f"m{i}")
        for i in range(40)
    ]
    answer_question(
        question="latest",
        page_id="pid",
        page_text="Some page body text here that is long enough." * 5,
        history=history,
    )
    sent = mock_post.call_args.kwargs["json"]
    # 2 system messages + bounded history (<= 12) + final user question.
    assert len(sent["messages"]) <= 2 + SETTINGS.qa_max_history_messages + 1
    assert sent["messages"][-1]["content"] == "latest"
