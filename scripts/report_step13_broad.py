"""Step 13 broad run (train 1979-2015): Korean PDF summary of the scores, logs and figures.

Reads runs/step13_broad/score_best.json (arm metrics + bootstrap vs seqregress_pinn) and embeds the two
figures in docs/step13_broad. The numbers quoted from the logs (raw / PINN base models, flow-dependence,
operator ablation) are copied from runs/step13_broad/*.log of the 10-27 run. Needs reportlab (not in
requirements.txt) and a TrueType font with Hangul (DroidSansFallback).

    python scripts/report_step13_broad.py --scores runs/step13_broad/score_best.json --out docs/step13_broad
"""

import argparse
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FONT_PATH = "/usr/share/fonts/google-droid/DroidSansFallback.ttf"
FONT = "KR"
INK, MUTED, GRID, ACCENT, HEAD_BG = "#0b0b0b", "#52514e", "#d9d8d4", "#e34948", "#f1f0ec"
ARMS = (
    ("seqregress_pinn", "seqregress + PINN"),
    ("seqfm_dct_pinn_regm_band", "regm_band"),
    ("seqfm_dct_pinn_regm_flow", "regm_flow"),
    ("seqfm_dct_pinn_regm_flowop", "regm_flowop"),
)
LEAD_IDX = (0, 2, 5)  # 24 / 72 / 144 h


def styles():
    base = dict(fontName=FONT, textColor=colors.HexColor(INK), leading=15, fontSize=10)
    return {
        "title": ParagraphStyle("title", **{**base, "fontSize": 18, "leading": 24, "alignment": TA_CENTER}),
        "sub": ParagraphStyle("sub", **{**base, "fontSize": 9.5, "textColor": colors.HexColor(MUTED),
                                        "alignment": TA_CENTER}),
        "h1": ParagraphStyle("h1", **{**base, "fontSize": 13.5, "leading": 20, "spaceBefore": 10, "spaceAfter": 4,
                                      "textColor": colors.HexColor(ACCENT)}),
        "body": ParagraphStyle("body", **base),
        "bullet": ParagraphStyle("bullet", **{**base, "leftIndent": 12, "bulletIndent": 2}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 8.5, "leading": 11}),
        "cap": ParagraphStyle("cap", **{**base, "fontSize": 8.5, "textColor": colors.HexColor(MUTED)}),
    }


def table(rows, widths, st, bold_rows=()):
    data = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(HEAD_BG)),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(INK)),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor(GRID)),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for r in bold_rows:
        style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fdeceb")))
    t.setStyle(TableStyle(style))
    return t


def bullets(items, st):
    return [Paragraph(s, st["bullet"], bulletText="•") for s in items]


def fmt3(vals, nd):
    return " / ".join(f"{v:.{nd}f}" if nd else f"{v:.0f}" for v in vals)


def figure(path, width):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(path).getSize()
    return Image(path, width=width, height=width * h / w)


def build(scores, out_dir):
    pdfmetrics.registerFont(TTFont(FONT, FONT_PATH))
    # the font has no bold face; <b> maps to the same face
    pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)
    st = styles()
    arms = scores["arms"]
    boot = scores["bootstrap_vs_seqregress_pinn"]
    full_w = A4[0] - 36 * mm
    story = []

    story += [
        Paragraph("Step 13 broad run 결과 요약", st["title"]),
        Spacer(1, 4),
        Paragraph("physical_guide_flowmatching · branch claude/project-thread-43gbu8 · 2026-10-27", st["sub"]),
        Spacer(1, 10),
    ]

    story.append(Paragraph("1. 실험 설정", st["h1"]))
    story += bullets([
        "데이터: 실제 ERA5를 5.625°로 보수적 regrid (fake data 미사용). 학습 1979–2015, 검증 2016, "
        "테스트 2017–2018 (ClimODE와 같은 360 IC), 앙상블 8 member.",
        "모델: PINN projection 8000 it (batch 8), seqfm_dct (flow matching) 40000 it (batch 16), "
        "seqregress (회귀 앙상블) 40000 it (batch 16) — GPU. Guide 적합과 채점은 CPU.",
        "Guide arm: <b>regm_band</b> = a + b·회귀 평균 + c·x(t) + FM 편차 + band Γ, "
        "<b>regm_flow</b> = regm_band + flow-dependent spread field, "
        "<b>regm_flowop</b> = regm_band + operator spread field.",
        "Run 종료: 2026-10-27 01:04 (본 실험 + operator ablation 모두 exit 0). 결과: runs/step13_broad/.",
    ], st)

    story.append(Paragraph("2. 핵심 결론", st["h1"]))
    story += bullets([
        "FM + PINN + regm guide가 이번 run의 최선. z500 144h RMSE 741로 회귀 앙상블(789)과 "
        "ClimODE 논문(784)보다 낮고, z500 144h CRPS는 회귀 대비 약 8% 개선 (0.0310 vs 0.0338).",
        "Bootstrap 95% CI: 세 guide arm 모두 <b>모든 변수 × 모든 lead</b>에서 CRPS가 seqregress_pinn보다 "
        "유의하게 낮음 (차이의 CI 상한이 전부 0 미만).",
        "세 arm 간 차이는 0.6% 이내. 전체 fair CRPS: band 0.1595, flow 0.1585, flowop 0.1585 — "
        "flowop은 flow 대비 이득 없음.",
        "Calibration: guide arm의 z500 SSR은 1.10–1.23으로 안정적 (회귀는 144h에서 1.33). "
        "단, 여전히 약간 과분산이고 day-6 하위 tercile BSS는 -0.17.",
        "Flow-dependence는 약함: spread–오차 상관 band ≈ 0.01, flow 0.08–0.10, flowop 0.04–0.07.",
    ], st)

    story.append(Paragraph("3. 기본 모델 (guide 이전, 테스트 2017–18, z500)", st["h1"]))
    rows = [
        ["모델", "RMSE 24 / 144h (m²/s²)", "CRPS 24 / 144h", "SSR 24 / 144h", "정역학 잔차"],
        ["seqfm_dct raw", "371 / 935", "0.0176 / 0.0415", "2.08 / 1.70", "7.97"],
        ["seqfm_dct + PINN", "372 / 933", "0.0176 / 0.0413", "2.08 / 1.69", "5.42"],
        ["seqregress raw", "170 / 790", "0.0077 / 0.0339", "1.25 / 1.34", "5.22"],
        ["seqregress + PINN", "170 / 789", "0.0077 / 0.0338", "1.25 / 1.33", "3.76"],
    ]
    story.append(table(rows, [38 * mm, 38 * mm, 32 * mm, 28 * mm, 26 * mm], st))
    story.append(Spacer(1, 3))
    story += bullets([
        "FM 단독은 회귀보다 크게 나쁘고 spread가 오차의 약 2배 (SSR ≈ 2).",
        "PINN projection은 정확도를 거의 바꾸지 않지만 물리 잔차, 특히 정역학 잔차를 줄임 "
        "(진짜 대기 잔차 기준 ≈ 1).",
    ], st)

    story.append(Paragraph("4. Guide arm vs 회귀 앙상블 (테스트 2017–18)", st["h1"]))
    rows = [["arm", "z500 RMSE<br/>24 / 72 / 144h", "t850 RMSE (K)<br/>24 / 72 / 144h",
             "t2m RMSE (K)<br/>24 / 72 / 144h", "z500 CRPS<br/>24 / 72 / 144h", "z500 SSR<br/>24 / 72 / 144h"]]
    for key, name in ARMS:
        a = arms[key]
        pick = lambda m, v, idx=LEAD_IDX: [a[m][i][v] for i in idx]  # noqa: E731
        rows.append([name, fmt3(pick("rmse", 0), 0), fmt3(pick("rmse", 1), 2), fmt3(pick("rmse", 2), 2),
                     fmt3(pick("crps", 0), 4), fmt3(pick("ssr", 0), 2)])
    story.append(table(rows, [24 * mm, 25 * mm, 27 * mm, 27 * mm, 38 * mm, 33 * mm], st, bold_rows=(2, 3)))
    story.append(Spacer(1, 3))

    hi_max = {k: max(max(r) for r in v["crps"]["hi"]) for k, v in boot.items()}
    story += bullets([
        "CRPS는 z500을 정규화한 값. 세 arm의 평균장은 공유되므로 RMSE가 동일.",
        "Bootstrap (vs seqregress_pinn) CRPS 차이의 95% CI 상한 최댓값: "
        + ", ".join(f"{k.replace('seqfm_dct_pinn_', '')} {v:.1e}" for k, v in hi_max.items())
        + " — 모두 음수, 즉 전 변수·전 lead에서 유의한 개선.",
        "작은 스케일 파워 (k13–32, 1이 완벽): z500 144h guide ≈ 5.0, 회귀 11.5. 덜 과하지만 여전히 과대.",
        "Doob ratio 1.00, falling steps 0 — 에너지가 lead에 따라 단조 증가 (물리적으로 일관).",
    ], st)

    story.append(Paragraph("5. 그림", st["h1"]))
    story.append(figure(os.path.join(out_dir, "fig_broad.png"), full_w))
    story.append(Paragraph(
        "그림 1. RMSE (위)와 CRPS / spread (아래): 회귀 앙상블 (초록), regm arm 3개 (빨강/주황/청록, 거의 겹침), "
        "persistence, ClimODE 논문 값.", st["cap"]))
    story.append(Spacer(1, 8))
    story.append(figure(os.path.join(out_dir, "fig_broad_vs_climode.png"), full_w))
    story.append(Paragraph(
        "그림 2. 같은 360 테스트 IC에서의 결정론적 RMSE: ClimODE 논문, 4년 학습 (2012–2015) step 13, "
        "이번 broad 학습 (1979–2015). 학습 기간 확장으로 모든 lead에서 개선.", st["cap"]))

    story.append(PageBreak())
    story.append(Paragraph("6. Flow-dependence (spread가 상황별 오차를 따라가는가)", st["h1"]))
    rows = [
        ["arm", "spread CV", "spread–|err| 상관", "spread bin (하→상)", "RMSE bin (하→상)"],
        ["seqregress_pinn", "0.27", "≈ 0.00", "0.64 → 1.40", "1.00 → 1.00"],
        ["regm_band", "0.27", "0.01", "0.63 → 1.40", "0.98 → 1.02"],
        ["regm_flow", "0.31–0.32", "0.08–0.11", "0.59 → 1.50", "0.88 → 1.15"],
        ["regm_flowop", "0.29–0.31", "0.04–0.07", "0.60 → 1.46", "0.92 → 1.09"],
    ]
    story.append(table(rows, [36 * mm, 24 * mm, 34 * mm, 34 * mm, 34 * mm], st))
    story.append(Spacer(1, 3))
    story += bullets([
        "z500 / t850, 24–144h 범위 (runs/step13_broad/flowdep.log). spread가 큰 case에서 실제 오차도 큰 정도는 "
        "flow > flowop > band 순이지만 절대값은 작음.",
        "Guide가 예측한 σ와 |error|의 case별 상관 (z500): flow_sigma 0.15–0.18, flow_op 0.12–0.16 (2017, 2018).",
    ], st)

    story.append(Paragraph("7. Operator ablation (결정론적 평균, 테스트 2017–18)", st["h1"]))
    rows = [
        ["평균장", "z500 RMSE 24 / 72 / 144h", "t850 RMSE 24 / 72 / 144h (K)"],
        ["mean (seqfm_dct_pinn)", "372 / 666 / 933", "2.03 / 3.15 / 3.99"],
        ["mbmx", "333 / 631 / 863", "1.92 / 3.00 / 3.72"],
        ["operator", "367 / 643 / 886", "2.00 / 3.01 / 3.81"],
        ["operator_mean_only", "334 / 600 / 810", "1.84 / 2.75 / 3.40"],
        ["persistence", "572 / 903 / 1025", "2.70 / 3.91 / 4.25"],
    ]
    story.append(table(rows, [50 * mm, 60 * mm, 60 * mm], st, bold_rows=(4,)))
    story.append(Spacer(1, 3))
    story += bullets([
        "FM 앙상블 위의 operator 중 operator_mean_only가 가장 좋지만, 회귀 평균 (z500 170 / 427 / 789)보다는 "
        "단기에서 크게 뒤짐. guide arm의 평균장이 회귀 평균을 쓰는 이유.",
    ], st)

    story.append(Paragraph("8. 다음 단계 제안", st["h1"]))
    story += bullets([
        "FM 자체의 과분산 (SSR ≈ 2)과 작은 스케일 과대 파워를 줄이는 것이 우선 — guide가 보정하는 몫을 줄임.",
        "flowop은 flow 대비 이득이 없으므로 단순한 regm_flow를 기본 arm으로 채택.",
        "Flow-dependence 강화: spread field 입력에 흐름 정보 (예: 회귀 앙상블 분산, 블로킹 지표)를 추가하는 실험.",
        "하위 tercile day-6 BSS 음수의 원인 (평균 편향 vs spread) 분해.",
    ], st)

    out = os.path.join(out_dir, "step13_broad_report.pdf")
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Step 13 broad run 결과 요약",
                            author="physical_guide_flowmatching")
    doc.build(story)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="runs/step13_broad/score_best.json")
    p.add_argument("--out", default="docs/step13_broad")
    a = p.parse_args()
    with open(a.scores) as f:
        scores = json.load(f)
    print(build(scores, a.out))


if __name__ == "__main__":
    main()
