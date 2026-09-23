"""PDF를 페이지 단위로 읽어서 청크로 나눈다. 각 청크는 출처 페이지 번호를 가진다."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # 파일명
    page: int  # 1부터 시작


def load_pdf(path: str | Path) -> list[tuple[int, str]]:
    """(페이지 번호, 텍스트) 리스트를 반환한다. 텍스트가 없는 페이지는 건너뛴다."""
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, " ".join(text.split())))
    return pages


def split_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """글자 수 기준으로 자르되, 가능하면 문장 끝(. ? !)에서 자른다."""
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            # 청크 뒤쪽 절반 안에서 마지막 문장 끝을 찾는다
            cut = max(text.rfind(p, start + chunk_size // 2, end) for p in (". ", "? ", "! "))
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pdf(path: str | Path, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    """청크가 페이지 경계를 넘지 않게 해서 출처 페이지를 정확히 표시할 수 있게 한다."""
    name = Path(path).name
    chunks = []
    for page_no, text in load_pdf(path):
        for j, piece in enumerate(split_text(text, chunk_size, overlap)):
            chunks.append(Chunk(id=f"{name}:p{page_no}:c{j}", text=piece, source=name, page=page_no))
    return chunks
