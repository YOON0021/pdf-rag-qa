# 📄 PDF RAG Q&A

PDF를 올리고 질문하면 **문서 내용만 근거로** 답변하고, **출처 페이지**를 함께 보여주는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

<!-- 데모 GIF를 여기에 넣으세요: ![demo](docs/demo.gif) -->

## 주요 기능

- PDF 여러 개 업로드 및 인덱싱
- 답변에 근거 번호 `[1]`, `[2]` 표시 + 출처 파일명·페이지·유사도 확인
- 문서에 없는 내용은 "찾을 수 없음"으로 답변 (환각 억제)
- 한국어 지원 로컬 임베딩 모델 사용 → 임베딩 비용 0원
- API 호출 없이 돌아가는 **검색 성능 평가 스크립트** (hit@k, MRR)

## 아키텍처

```
[인덱싱]  PDF ─► 페이지별 텍스트 추출 ─► 청크 분할(페이지 경계 유지) ─► 임베딩 ─► Chroma DB

[질의]    질문 ─► 임베딩 ─► Chroma top-k 검색 ─► 프롬프트(context + 질문) ─► Claude ─► 답변 + 출처
```

| 구성 요소 | 사용 기술 |
|---|---|
| PDF 파싱 | pypdf |
| 임베딩 | `intfloat/multilingual-e5-small` (sentence-transformers) |
| 벡터 DB | Chroma (로컬 저장) |
| LLM | Claude (Anthropic API) |
| UI | Streamlit |

## 실행 방법

```bash
git clone https://github.com/YOON0021/pdf-rag-qa.git
cd pdf-rag-qa
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # ANTHROPIC_API_KEY 입력
streamlit run app.py
```

처음 실행할 때 임베딩 모델(약 470MB)을 자동으로 내려받습니다.

## 검색 성능 평가

샘플 PDF(가상 회사의 취업규칙 6페이지)와 질문 14개가 포함되어 있어서 바로 실행할 수 있습니다.

```bash
python -m eval.run_eval --chunk-sizes 100 200 800
```

| chunk_size | chunks | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|---|
| 100 | 16 | 0.93 | 0.93 | 1.00 | 0.95 |
| 200 | 6 | 0.93 | 1.00 | 1.00 | 0.96 |
| 800 | 6 | 0.93 | 1.00 | 1.00 | 0.96 |

내 PDF로 평가하려면 `eval/questions.json`에 질문과 정답 페이지를 적고 실행합니다.

```bash
python -m eval.run_eval --pdf data/my.pdf --questions eval/questions.json
```

샘플 PDF를 다시 만들려면 `pip install -r requirements-dev.txt` 후 `python scripts/make_sample_pdf.py`를 실행합니다.

- **hit@k**: 정답 페이지가 상위 k개 검색 결과 안에 들어온 비율
- **MRR**: 정답이 처음 나온 순위의 역수 평균 (1에 가까울수록 좋음)

## 프로젝트 구조

```
├── app.py              # Streamlit UI
├── rag/
│   ├── loader.py       # PDF 로드 + 청크 분할
│   ├── store.py        # 임베딩 + Chroma 저장/검색
│   └── llm.py          # 프롬프트 구성 + Claude 호출 (스트리밍)
├── eval/
│   ├── run_eval.py             # 검색 성능 평가
│   ├── sample.pdf              # 평가용 샘플 문서
│   ├── sample_questions.json   # 샘플 문서용 질문셋
│   └── questions.json          # 내 문서용 질문셋 템플릿
└── scripts/
    └── make_sample_pdf.py      # 샘플 PDF 생성
```

## 앞으로 개선할 점

- [ ] BM25와 벡터 검색을 합친 하이브리드 검색
- [ ] Reranker 적용
- [ ] 표·이미지가 많은 PDF 처리 (OCR)
- [ ] 멀티턴 대화 (이전 질문 맥락 반영)
