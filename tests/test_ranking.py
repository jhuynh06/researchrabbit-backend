import numpy as np
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from app.ranking import rank_chunks


def _mock_embed_response(texts):
    """Return a mock httpx response with fake embeddings."""
    dim = 2
    embeddings = [[float(i) / max(len(texts), 1), 1.0 - float(i) / max(len(texts), 1)] for i in range(len(texts))]
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "data": [{"embedding": embeddings[i], "index": i, "object": "embedding"} for i in range(len(texts))],
        "model": "qwen3-embedding-0.6b",
        "object": "list",
    }
    return mock_response


def test_rank_chunks_sorts_by_similarity(monkeypatch) -> None:
    # First call: prompt + 2 chunks. Second call: sentences from the top chunks
    # (for sentence-level refinement). Score the "dataset" sentence as the
    # closest match to the prompt vector [1.0, 0.0].
    def fake_embed_texts(texts: list[str]) -> np.ndarray:
        if texts and texts[0] == "dataset":
            return np.array(
                [
                    [1.0, 0.0],
                    [0.1, 0.9],
                    [0.9, 0.1],
                ]
            )
        # Sentence-refinement call: one row per sentence, score by keyword.
        return np.array(
            [[0.9, 0.1] if "dataset" in t else [0.1, 0.9] for t in texts]
        )

    monkeypatch.setattr("app.ranking.embed_texts", fake_embed_texts)
    monkeypatch.setattr(
        "app.ranking.get_settings",
        lambda: SimpleNamespace(
            max_chunk_words=5,
            overlap_words=0,
            default_top_k=5,
            max_top_k=10,
            candidate_explanation="Highest semantic match to the prompt.",
            qa_anchor_words=8,
        ),
    )
    page_text = "Background information about theory.\n\nThe dataset contains clinical records."

    chunks = rank_chunks(page_text=page_text, prompt="dataset", top_k=2)

    assert chunks[0].text == "The dataset contains clinical records."
    assert chunks[0].score > chunks[1].score


@patch("app.embeddings.httpx.post")
def test_embed_texts_calls_do_api(mock_post, monkeypatch) -> None:
    monkeypatch.setattr("app.embeddings.get_settings", lambda: SimpleNamespace(
        do_inference_token="test-token",
        do_inference_url="https://inference.do-ai.run/v1/embeddings",
        embedding_model="qwen3-embedding-0.6b",
        embedding_timeout=10.0,
    ))
    mock_post.return_value = _mock_embed_response(["hello", "world"])

    from app.embeddings import embed_texts
    result = embed_texts(["hello", "world"])

    assert result.shape == (2, 2)
    mock_post.assert_called_once()
