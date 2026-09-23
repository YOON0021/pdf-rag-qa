"""검색된 청크를 근거로 Claude에게 답변을 요청한다."""

import os
from collections.abc import Iterator

import anthropic

from rag.store import Hit

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")

SYSTEM_PROMPT = """당신은 사용자가 올린 문서에 대한 질문에 답하는 도우미입니다.
- 반드시 제공된 <context> 안의 내용만 근거로 답하세요.
- 답변 속 주장마다 근거 문단 번호를 [1], [2]처럼 표시하세요.
- context에 답이 없으면 "문서에서 해당 내용을 찾을 수 없습니다."라고 답하고 추측하지 마세요.
- 사용자가 질문한 언어로 답하세요."""


def build_context(hits: list[Hit]) -> str:
    parts = [
        f'<doc index="{i}" source="{h.source}" page="{h.page}">\n{h.text}\n</doc>'
        for i, h in enumerate(hits, start=1)
    ]
    return "<context>\n" + "\n".join(parts) + "\n</context>"


def answer_stream(question: str, hits: list[Hit]) -> Iterator[str]:
    client = anthropic.Anthropic()
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},  # 문서 기반 Q&A는 low로도 충분하고 빠르다
        # 안전 분류기가 요청을 거절하면 서버에서 다른 모델로 자동 재시도
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": f"{build_context(hits)}\n\n질문: {question}"}],
    ) as stream:
        yield from stream.text_stream
        final = stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n(모델이 이 요청에 대한 답변을 거절했습니다.)"
    elif final.stop_reason == "max_tokens":
        yield "\n\n(답변이 길어서 중간에 잘렸습니다.)"
