# 📄 PDF RAG Q&A

PDF를 올리고 질문하면 **문서 내용만 근거로** 답변하고, **출처 페이지**를 함께 보여주는 RAG(Retrieval-Augmented Generation) 챗봇입니다.

<!-- 데모 GIF를 여기에 넣으세요: ![demo](docs/demo.gif) -->

## 주요 기능

- PDF 여러 개 업로드 및 인덱싱
- 답변에 근거 번호 `[1]`, `[2]` 표시 + 출처 파일명·페이지·유사도 확인
- 문서에 없는 내용은 "찾을 수 없음"으로 답변 (환각 억제)
- 한국어 지원 로컬 임베딩 모델 사용 → 임베딩 비용 0원
- **하이브리드 검색**: 벡터 검색 + BM25 키워드 검색을 RRF(Reciprocal Rank Fusion)로 결합
- API 호출 없이 돌아가는 **검색 성능 평가 스크립트** (hit@k, MRR)

## 아키텍처

```
[인덱싱]  PDF ─► 페이지별 텍스트 추출 ─► 청크 분할(페이지 경계 유지) ─► 임베딩 ─► Chroma DB

[질의]    질문 ─┬► 벡터 검색 (Chroma) ─┬► RRF로 순위 결합 ─► top-k ─► 프롬프트 ─► Claude ─► 답변 + 출처
                └► BM25 키워드 검색 ───┘
```

BM25는 한국어 조사 문제("연차는" ≠ "연차")를 피하려고, 한글 단어를 글자 2개 단위(bigram)로도 쪼개서 색인합니다. 형태소 분석기 없이 부분 일치를 잡는 간단한 방법입니다.

| 구성 요소 | 사용 기술 |
|---|---|
| PDF 파싱 | pypdf |
| 임베딩 | `intfloat/multilingual-e5-small` (sentence-transformers) |
| 벡터 DB | Chroma (로컬 저장) |
| 키워드 검색 | rank-bm25 |
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

| chunk_size | chunks | mode | hit@1 | hit@3 | hit@5 | mrr |
|---|---|---|---|---|---|---|
| 100 | 16 | vector | 0.93 | 0.93 | 1.00 | 0.95 |
| 100 | 16 | hybrid | 0.93 | 0.93 | 1.00 | 0.95 |
| 200 | 6 | vector | 0.93 | 1.00 | 1.00 | 0.96 |
| 200 | 6 | hybrid | 0.93 | 1.00 | 1.00 | 0.96 |
| 800 | 6 | vector | 0.93 | 1.00 | 1.00 | 0.96 |
| 800 | 6 | hybrid | 0.93 | 1.00 | 1.00 | 0.96 |

**분석**: 샘플 문서가 6페이지로 작아서 두 방식의 차이가 나타나지 않았습니다. 유일하게 틀린 질문("아파서 4일 쉬려면 뭘 내야 해?" → 정답은 '병가' 조항)은 질문과 문서에 겹치는 단어가 없어서 BM25로도 찾을 수 없는 경우입니다. 하이브리드 검색은 제품명, 조항 번호, 코드처럼 **정확한 키워드**가 중요한 문서에서 효과가 크기 때문에, 더 크고 다양한 문서로 추가 비교가 필요합니다.

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
│   ├── store.py        # 임베딩 + Chroma 저장, 벡터/BM25/하이브리드 검색
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

- [x] BM25와 벡터 검색을 합친 하이브리드 검색
- [ ] Reranker 적용
- [ ] 표·이미지가 많은 PDF 처리 (OCR)
- [ ] 멀티턴 대화 (이전 질문 맥락 반영)
