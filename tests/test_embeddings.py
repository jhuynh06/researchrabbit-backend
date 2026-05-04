from unittest.mock import patch, MagicMock
from types import SimpleNamespace

import httpx
import pytest

from app.embeddings import embed_texts


SETTINGS = SimpleNamespace(
    do_inference_token="test-token",
    do_inference_url="https://inference.do-ai.run/v1/embeddings",
    embedding_model="qwen3-embedding-0.6b",
    embedding_timeout=10.0,
)


@patch("app.embeddings.get_settings", return_value=SETTINGS)
@patch("app.embeddings.httpx.post")
def test_embed_texts_returns_normalized_vectors(mock_post, mock_settings):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [
            {"embedding": [3.0, 4.0], "index": 0, "object": "embedding"},
            {"embedding": [0.0, 5.0], "index": 1, "object": "embedding"},
        ]
    }
    mock_post.return_value = mock_response

    result = embed_texts(["a", "b"])

    assert result.shape == (2, 2)
    assert abs(float((result[0] ** 2).sum()) - 1.0) < 1e-5
    assert abs(float((result[1] ** 2).sum()) - 1.0) < 1e-5


@patch("app.embeddings.get_settings", return_value=SETTINGS)
@patch("app.embeddings.httpx.post", side_effect=httpx.TimeoutException("timeout"))
def test_embed_texts_raises_on_timeout(mock_post, mock_settings):
    with pytest.raises(Exception) as exc_info:
        embed_texts(["hello"])
    assert exc_info.value.status_code == 502


@patch("app.embeddings.get_settings", return_value=SETTINGS)
@patch("app.embeddings.httpx.post")
def test_embed_texts_raises_on_api_error(mock_post, mock_settings):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_post.return_value = mock_response

    with pytest.raises(Exception) as exc_info:
        embed_texts(["hello"])
    assert exc_info.value.status_code == 502
