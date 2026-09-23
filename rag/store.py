"""임베딩 + Chroma 벡터 저장소, BM25와 합친 하이브리드 검색."""

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag.loader import Chunk

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
DB_PATH = os.getenv("CHROMA_PATH", ".chroma")
RRF_K = 60  # Reciprocal Rank Fusion 상수 (논문 기본값)


@dataclass
class Hit:
    text: str
    source: str
    page: int
    score: float  # vector: 코사인 유사도, hybrid: RRF 점수 (둘 다 높을수록 관련 있음)
    id: str = ""


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL)


def _embed(texts: list[str], kind: str) -> list[list[float]]:
    # e5 계열 모델은 "query: " / "passage: " 접두어를 붙여야 성능이 제대로 나온다
    if "e5" in EMBED_MODEL:
        texts = [f"{kind}: {t}" for t in texts]
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()


def tokenize(text: str) -> list[str]:
    """BM25용 토크나이저.

    한국어는 조사가 붙어서 "연차는"과 "연차"가 다른 단어가 되므로,
    한글이 들어간 단어는 글자 2개씩 자른 bigram도 함께 토큰으로 쓴다.
    (형태소 분석기 없이도 부분 일치를 잡기 위한 간단한 방법)
    """
    tokens = []
    for word in re.findall(r"\w+", text.lower()):
        tokens.append(word)
        if len(word) > 2 and re.search(r"[가-힣]", word):
            tokens += [word[i : i + 2] for i in range(len(word) - 1)]
    return tokens


class VectorStore:
    def __init__(self, collection: str = "docs", path: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self._bm25 = None  # 문서가 바뀌면 None으로 되돌려 다음 검색 때 다시 만든다

    def add(self, chunks: list[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            self.col.upsert(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=_embed([c.text for c in batch], "passage"),
                metadatas=[{"source": c.source, "page": c.page} for c in batch],
            )
        self._bm25 = None

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[Hit]:
        """mode: "vector" (임베딩만) 또는 "hybrid" (임베딩 + BM25를 RRF로 결합)."""
        count = self.col.count()
        if count == 0:
            return []
        if mode == "vector":
            return self._vector_search(query, min(k, count))

        # 두 검색기에서 후보를 넉넉히 뽑은 뒤 순위를 합친다
        n = min(max(k * 3, 10), count)
        ranked_lists = [self._vector_search(query, n), self._bm25_search(query, n)]
        fused: dict[str, Hit] = {}
        scores: dict[str, float] = {}
        for hits in ranked_lists:
            for rank, h in enumerate(hits, start=1):
                fused.setdefault(h.id, h)
                scores[h.id] = scores.get(h.id, 0.0) + 1 / (RRF_K + rank)

        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [Hit(fused[i].text, fused[i].source, fused[i].page, scores[i], i) for i in top]

    def _vector_search(self, query: str, n: int) -> list[Hit]:
        res = self.col.query(query_embeddings=_embed([query], "query"), n_results=n)
        return [
            Hit(text=doc, source=meta["source"], page=meta["page"], score=1 - dist, id=id_)
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]

    def _bm25_search(self, query: str, n: int) -> list[Hit]:
        if self._bm25 is None:
            data = self.col.get(include=["documents", "metadatas"])
            corpus = [tokenize(d) for d in data["documents"]]
            self._bm25 = (BM25Okapi(corpus), data)
        bm25, data = self._bm25

        scores = bm25.get_scores(tokenize(query))
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n]
        return [
            Hit(
                text=data["documents"][i],
                source=data["metadatas"][i]["source"],
                page=data["metadatas"][i]["page"],
                score=float(scores[i]),
                id=data["ids"][i],
            )
            for i in top
            if scores[i] > 0
        ]

    def sources(self) -> list[str]:
        metas = self.col.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metas})

    def reset(self) -> None:
        name = self.col.name
        self.client.delete_collection(name)
        self.col = self.client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        self._bm25 = None
