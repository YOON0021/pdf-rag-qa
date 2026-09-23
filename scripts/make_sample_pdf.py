"""평가용 샘플 PDF(가상의 회사 취업규칙)를 만든다.

사용법:
    pip install fpdf2
    python scripts/make_sample_pdf.py                # eval/sample.pdf 생성
    python scripts/make_sample_pdf.py --font /path/to/NanumGothic.ttf
"""

import argparse
from pathlib import Path

from fpdf import FPDF

# 한글 글리프가 있는 TTF 폰트 후보 (macOS, Ubuntu 순)
FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]

PAGES = [
    "제1장 회사 소개\n"
    "에코테크는 2015년 서울에서 설립된 친환경 배터리 재활용 기업이다. 본사는 경기도 성남시 판교에 있으며, "
    "충북 오창에 재활용 공장을 운영하고 있다. 2025년 기준 직원 수는 약 320명이다.",
    "제2장 근무 시간\n"
    "기본 근무 시간은 오전 9시부터 오후 6시까지이며 점심시간은 12시부터 1시까지다. "
    "시차출퇴근제를 신청하면 오전 8시에서 10시 사이에 자유롭게 출근할 수 있다. "
    "연장 근무는 사전에 팀장 승인을 받아야 하며, 주 12시간을 넘을 수 없다.",
    "제3장 휴가\n"
    "정규직 직원은 입사 첫해에 15일의 연차를 받는다. 3년 이상 근속하면 2년마다 1일씩 추가되며 최대 25일까지 가능하다. "
    "병가는 연 10일까지 유급으로 사용할 수 있고, 3일 이상 연속으로 쓸 경우 진단서를 제출해야 한다. "
    "본인 결혼 시 경조휴가 5일이 주어진다.",
    "제4장 재택근무\n"
    "직원은 주 2회까지 재택근무를 신청할 수 있다. 신청은 전주 금요일까지 팀장에게 사내 메신저로 해야 한다. "
    "신입사원은 수습 기간인 입사 3개월이 지난 뒤부터 재택근무가 가능하다. "
    "재택근무 중에도 오전 10시부터 오후 4시까지는 메신저 응답이 가능해야 한다.",
    "제5장 복리후생\n"
    "점심 식대로 월 20만원이 지급된다. 자기계발비는 연 100만원 한도로 도서 구입, 온라인 강의 수강, "
    "자격증 응시료에 사용할 수 있다. 매년 1회 종합 건강검진을 회사 비용으로 받을 수 있으며, "
    "배우자 검진비의 50%도 지원한다.",
    "제6장 보안\n"
    "회사 노트북에는 승인된 소프트웨어만 설치할 수 있다. 외부 USB 저장장치 사용은 금지되며, "
    "고객 데이터를 개인 이메일이나 클라우드로 전송해서는 안 된다. 노트북을 분실하면 24시간 이내에 보안팀에 신고해야 한다.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", help="한글을 지원하는 .ttf 폰트 경로")
    ap.add_argument("--out", default="eval/sample.pdf")
    args = ap.parse_args()

    font = args.font or next((f for f in FONT_CANDIDATES if Path(f).exists()), None)
    if font is None:
        raise SystemExit("한글 폰트를 찾지 못했습니다. --font 로 .ttf 경로를 지정하세요.")

    pdf = FPDF()
    pdf.add_font("kr", fname=font)
    pdf.set_font("kr", size=12)
    for text in PAGES:
        pdf.add_page()
        pdf.multi_cell(0, 8, text)
    pdf.output(args.out)
    print(f"{args.out} 생성 완료 ({len(PAGES)}페이지)")


if __name__ == "__main__":
    main()
