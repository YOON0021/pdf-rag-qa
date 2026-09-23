"""Streamlit UI: PDF 업로드 → 인덱싱 → 질문 → 출처와 함께 답변."""

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from rag.llm import answer_stream  # noqa: E402
from rag.loader import chunk_pdf  # noqa: E402
from rag.store import VectorStore  # noqa: E402

st.set_page_config(page_title="PDF RAG Q&A", page_icon="📄")
st.title("📄 PDF에게 물어보기")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


store = get_store()

with st.sidebar:
    st.header("문서")
    files = st.file_uploader("PDF 업로드", type="pdf", accept_multiple_files=True)
    if files and st.button("인덱싱", type="primary"):
        for f in files:
            with st.spinner(f"{f.name} 처리 중..."):
                tmp = Path(tempfile.mkdtemp()) / f.name
                tmp.write_bytes(f.getvalue())
                chunks = chunk_pdf(tmp)
                store.add(chunks)
            st.success(f"{f.name}: {len(chunks)}개 청크")

    indexed = store.sources()
    st.caption("인덱싱된 문서: " + (", ".join(indexed) if indexed else "없음"))
    k = st.slider("검색할 청크 수 (top-k)", 1, 10, 4)
    mode = st.radio(
        "검색 방식",
        ["hybrid", "vector"],
        format_func={"hybrid": "하이브리드 (벡터 + BM25)", "vector": "벡터만"}.get,
    )
    if indexed and st.button("전체 초기화"):
        store.reset()
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("문서에 대해 질문하세요"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        hits = store.search(question, k=k, mode=mode)
        if not hits:
            answer = "먼저 사이드바에서 PDF를 업로드하고 인덱싱해 주세요."
            st.markdown(answer)
        else:
            answer = st.write_stream(answer_stream(question, hits))
            with st.expander("📚 출처"):
                for i, h in enumerate(hits, start=1):
                    st.markdown(f"**[{i}] {h.source} · p.{h.page}** (점수 {h.score:.3f})")
                    st.caption(h.text[:400] + ("…" if len(h.text) > 400 else ""))
    st.session_state.messages.append({"role": "assistant", "content": answer})
