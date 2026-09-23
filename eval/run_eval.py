"""검색(retrieval) 성능 평가: 정답 페이지가 top-k 안에 들어오는지 측정한다.

LLM을 호출하지 않으므로 API 비용이 들지 않는다.
청크 크기 / 오버랩 / 임베딩 모델을 바꿔가며 결과를 비교해 보자.

사용법:
    python -m eval.run_eval                          # 샘플 PDF + 샘플 질문셋
    python -m eval.run_eval --chunk-sizes 100 200 800
    python -m eval.run_eval --pdf data/my.pdf --questions eval/questions.json
"""

import argparse
import json
import tempfile

from rag.loader import chunk_pdf
from rag.store import EMBED_MODEL, VectorStore


def evaluate(pdf: str, questions: list[dict], chunk_size: int, overlap: int, ks: list[int]) -> dict:
    store = VectorStore(collection="eval", path=tempfile.mkdtemp())
    chunks = chunk_pdf(pdf, chunk_size, overlap)
    store.add(chunks)

    hits_at = {k: 0 for k in ks}
    mrr = 0.0
    for q in questions:
        pages = [h.page for h in store.search(q["question"], k=max(ks))]
        answer_pages = set(q["pages"])
        rank = next((i for i, p in enumerate(pages, start=1) if p in answer_pages), None)
        if rank:
            mrr += 1 / rank
        for k in ks:
            hits_at[k] += rank is not None and rank <= k

    n = len(questions)
    return {
        "chunk_size": chunk_size,
        "chunks": len(chunks),
        **{f"hit@{k}": hits_at[k] / n for k in ks},
        "mrr": mrr / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="eval/sample.pdf")
    ap.add_argument("--questions", default="eval/sample_questions.json")
    ap.add_argument("--chunk-sizes", type=int, nargs="+", default=[800])
    ap.add_argument("--overlap", type=int, default=150)
    ap.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    with open(args.questions, encoding="utf-8") as f:
        questions = json.load(f)

    print(f"임베딩 모델: {EMBED_MODEL} | 질문 {len(questions)}개\n")
    rows = [evaluate(args.pdf, questions, cs, min(args.overlap, cs // 2), args.k) for cs in args.chunk_sizes]

    # README에 바로 붙여넣을 수 있는 마크다운 표
    cols = list(rows[0].keys())
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in r.values()) + " |")


if __name__ == "__main__":
    main()
