import hashlib
from collections import OrderedDict

import numpy as np

MAX_CACHE_SIZE = 200


class EmbeddingCache:
    def __init__(self, max_size: int = MAX_CACHE_SIZE):
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_size = max_size

    def _key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, page_text: str) -> np.ndarray | None:
        key = self._key(page_text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, page_text: str, embeddings: np.ndarray) -> None:
        key = self._key(page_text)
        self._cache[key] = embeddings
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


cache = EmbeddingCache()
