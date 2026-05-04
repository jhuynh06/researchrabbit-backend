import httpx
import numpy as np
from fastapi import HTTPException

from app.config import get_settings


def embed_texts(texts: list[str]) -> np.ndarray:
    settings = get_settings()
    try:
        response = httpx.post(
            settings.do_inference_url,
            headers={"Authorization": f"Bearer {settings.do_inference_token}"},
            json={"model": settings.embedding_model, "input": texts, "encoding_format": "float"},
            timeout=settings.embedding_timeout,
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=502, detail="Embedding service timed out.")

    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {response.status_code}")

    data = sorted(response.json()["data"], key=lambda d: d["index"])
    vectors = np.array([d["embedding"] for d in data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)
