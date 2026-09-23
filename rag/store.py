"""임베딩 + Chroma 벡터 저장소."""

import os
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from rag.loader import Chunk

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
DB_PATH = os.getenv("CHROMA_PATH", ".chroma")


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # 코사인 유사도 (높을수록 비슷함)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def _embed(texts: list[str], kind: str) -> list[list[float]]:
    # e5 계열 모델은 "query: " / "passage: " 접두어를 붙여야 성능이 제대로 나온다
    if "e5" in EMBED_MODEL:
        texts = [f"{kind}: {t}" for t in texts]
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


class VectorStore:
    def __init__(self, collection: str = "docs", path: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=_embed([c.text for c in batch], "passage"),
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
            )

    def search(self, query: str, k: int = 4) -> list[Hit]:
        if self.col.count() == 0:
            return []
        res = self.col.query(query_embeddings=_embed([query], "query"), n_results=min(k, self.col.count()))
        return [
            Hit(text=doc, source=meta["source"], page=meta["page"], score=1 - dist)
            for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def sources(self) -> list[str]:
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def reset(self) -> None:
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
