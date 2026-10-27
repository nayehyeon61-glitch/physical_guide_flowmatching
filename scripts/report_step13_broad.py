"""Step 13 broad run: Korean PDF summary read from the run directory's JSON outputs (nothing copied from logs).

Inputs under --runs (a missing file drops its section, the PDF says so):
  score_best.json                score_step13.py: arms (crps/rmse/ssr/..., leads x vars), bootstrap vs the reference
  seqfm_dct_pinn.json            stage 2 FM: args, "test" (raw / pinn arms, truth residual), run record, best_it
  seqregress_pinn.json           stage 2 regression ensemble, same layout
  pinn_project.json              stage 1 PINN: args, run record, train_seconds
  flowdep.json                   diag_step13_flowdep.py: spread-error diagnostics per arm
  operator_seqfm_dct_pinn.json   operator ablation on the FM ensemble (deterministic means), optional
  <tag>.pt_metrics.csv           training curves written by pgcfm.runs.fit -> fig_training.png in --out
Figures made by plot_step13_broad.py (fig_broad.png, fig_broad_vs_climode.png) are embedded when they exist
in --out. Needs reportlab (not in requirements.txt) and a TrueType font with Hangul (FONTS, first found).

    python scripts/report_step13_broad.py --runs runs/step13_broad --out docs/step13_broad
"""

import argparse
import csv
import datetime
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

FONTS = (  # TrueType outlines with Hangul; the first that exists is used (a .ttc takes its first face)
    "/usr/share/fonts/google-droid/DroidSansFallback.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
)
FONT = "KR"
INK, MUTED, GRID, ACCENT, HEAD_BG = "#0b0b0b", "#52514e", "#d9d8d4", "#e34948", "#f1f0ec"
REF = "seqregress_pinn"
ARMS = (  # score arms in the order of the table; absent ones are skipped
    (REF, "seqregress + PINN"),
    ("seqfm_dct_pinn_regm_band", "regm_band"),
    ("seqfm_dct_pinn_regm_flow", "regm_flow"),
    ("seqfm_dct_pinn_regm_flowop", "regm_flowop"),
    ("seqfm_dct_pinn_regm_sband_clim_flow", "regm_sband_clim_flow"),
)
STAGES = (("pinn_project", "PINN projection"), ("seqfm_dct", "seqfm_dct (flow matching)"),
          ("seqregress", "seqregress (regression ensemble)"))
SHOW_H = (24, 72, 144)
VAR_NAMES = {"z": "z500", "t": "t850", "t2m": "t2m", "u10": "u10", "v10": "v10"}
NEXT_STEPS = (  # the owner's proposals after the 2026-10-27 run (judgement, not numbers)
    "FM 자체의 과분산 (SSR ≈ 2)과 작은 스케일 과대 파워를 줄이는 것이 우선 — guide가 보정하는 몫을 줄임.",
    "flowop은 flow 대비 이득이 없으므로 단순한 regm_flow를 기본 arm으로 채택.",
    "Flow-dependence 강화: spread field 입력에 흐름 정보 (예: 회귀 앙상블 분산, 블로킹 지표)를 추가하는 실험.",
    "하위 tercile day-6 BSS 음수의 원인 (평균 편향 vs spread) 분해.",
)


def register_font():
    for path in FONTS:
        if os.path.exists(path):
            pdfmetrics.registerFont(TTFont(FONT, path, subfontIndex=0))
            pdfmetrics.registerFontFamily(FONT, normal=FONT, bold=FONT, italic=FONT, boldItalic=FONT)
            return path
    raise FileNotFoundError("no Hangul TrueType font found; install one of " + ", ".join(FONTS))


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


def rng_text(vals, nd=2):
    """'0.27' when the values agree to the last digit, else '0.29–0.31'."""
    lo, hi = min(vals), max(vals)
    return f"{lo:.{nd}f}" if f"{lo:.{nd}f}" == f"{hi:.{nd}f}" else f"{lo:.{nd}f}–{hi:.{nd}f}"


def figure(path, width):
    from reportlab.lib.utils import ImageReader

    w, h = ImageReader(path).getSize()
    return Image(path, width=width, height=width * h / w)


def load(runs, name):
    path = os.path.join(runs, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def years_text(ys):
    ys = sorted(int(y) for y in ys)
    return f"{ys[0]}–{ys[-1]}" if len(ys) > 1 and ys[-1] - ys[0] == len(ys) - 1 else ", ".join(map(str, ys))


def run_rows(stages):
    """One row per stage: iterations, batch, device, commit, hours, best iteration, how it stopped."""
    rows = [["단계", "it × batch", "device", "commit", "학습 시간", "best it / 종료"]]
    for tag, name in STAGES:
        r = stages.get(tag)
        if r is None:
            rows.append([name, "-", "-", "-", "-", "(json 없음)"])
            continue
        ar, run = r.get("args", {}), r.get("run") or {}
        secs = r.get("train_seconds")
        commit = str(run.get("commit", "-"))[:8] + (" (dirty)" if run.get("dirty") else "")
        best = r.get("best_it")
        rows.append([name, f"{ar.get('iters', '-')} × {ar.get('batch', '-')}",
                     str(run.get("device_name", run.get("device", "기록 없음"))), commit,
                     f"{secs / 3600:.2f} h" if isinstance(secs, (int, float)) else "-",
                     (f"{best} / " if best is not None else "") + str(r.get("stopped", "-"))])
    return rows


def training_figure(runs, out_dir, stages):
    """fig_training.png from <tag>.pt_metrics.csv (pgcfm.runs.fit): train loss (and validation loss where
    it was measured) over the iterations of each stage, the selected iteration marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    found = []
    for tag, name in STAGES:
        path = os.path.join(runs, f"{tag}.pt_metrics.csv")
        if os.path.exists(path):
            with open(path) as f:
                found.append((tag, name, list(csv.DictReader(f))))
    if not found:
        return None
    fig, axes = plt.subplots(1, len(found), figsize=(5.2 * len(found), 3.8), layout="constrained", squeeze=False)
    for ax, (tag, name, rows) in zip(axes[0], found):
        tr = [(int(r["it"]), float(r["train_loss"])) for r in rows if r["train_loss"]]
        va = [(int(r["it"]), float(r["val_loss"])) for r in rows if r["val_loss"]]
        if tr:
            ax.plot(*zip(*tr), color="#2a78d6", lw=2, label="train (mean of the log window)")
        if va:
            ax.plot(*zip(*va), color="#e34948", lw=2, marker="o", ms=4, label="validation")
        best = (stages.get(tag) or {}).get("best_it")
        if best is not None and best >= 0:
            ax.axvline(best, color=MUTED, lw=1, ls=":", label=f"selected it {best}")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("iteration")
        ax.set_ylabel("loss")
        ax.grid(True, color="#e4e3df", lw=0.8)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(fontsize=8, frameon=False)
    path = os.path.join(out_dir, "fig_training.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def build(runs, out_dir, scores_name="score_best.json"):
    register_font()
    os.makedirs(out_dir, exist_ok=True)
    st = styles()
    full_w = A4[0] - 36 * mm
    scores = load(runs, scores_name)
    stages = {tag: load(runs, f"{tag}.json") for tag, _ in STAGES}
    fm, reg = stages["seqfm_dct"], stages["seqregress"]
    for tag in ("seqfm_dct", "seqregress"):  # stage 2 writes <tag>_pinn.json when it ran through the PINN
        if stages[tag] is None:
            stages[tag] = load(runs, f"{tag}_pinn.json")
    fm, reg = stages["seqfm_dct"], stages["seqregress"]
    flowdep, operator = load(runs, "flowdep.json"), load(runs, "operator_seqfm_dct_pinn.json")
    paper = load(os.path.join(os.path.dirname(__file__), "..", "docs", "step11"), "climode_paper_tables.json")
    paper_z = {int(h): v["z"] for h, v in paper["rmse"]["ClimODE"].items()} if paper else {}
    missing = []
    story = []

    story += [
        Paragraph("Step 13 broad run 결과 요약", st["title"]),
        Spacer(1, 4),
        Paragraph(f"physical_guide_flowmatching · {os.path.basename(os.path.abspath(runs))} · "
                  f"{datetime.date.today().isoformat()}", st["sub"]),
        Spacer(1, 10),
    ]

    # -- 1. setup, from the stage arguments and run records ------------------------------------------
    story.append(Paragraph("1. 실험 설정 (각 단계 json의 args / run 기록)", st["h1"]))
    src = fm or reg
    if src:
        ar = src["args"]
        members = ar.get("members", "-")
        items = [f"데이터: 실제 ERA5 5.625°. 학습 {years_text(ar['train_years'])}, 검증 {ar['val_year']}, "
                 f"테스트 {years_text(ar['test_years'])}, 앙상블 {members} member, K = {ar.get('K', '-')} (6 h 간격)."]
        for tag, name in STAGES:
            r = stages.get(tag)
            if r:
                a_, run = r["args"], r.get("run") or {}
                dev = run.get("device_name") or ("기록 없음 (pgcfm.runs 이전 버전)" if not run else run.get("device"))
                items.append(f"{name}: {a_.get('iters', '-')} it, batch {a_.get('batch', '-')}, lr {a_.get('lr', '-')}, "
                             f"device {dev}" + (f", 검증 매 {a_['val_every']} it" if a_.get("val_every") else ""))
        story += bullets(items, st)
    else:
        missing.append("seqfm_dct_pinn.json / seqregress_pinn.json")
        story += bullets(["단계 json이 없어 설정을 읽지 못함."], st)

    # -- 2. conclusions, filled from the score json ----------------------------------------------------
    story.append(Paragraph("2. 핵심 결론", st["h1"]))
    if scores:
        arms, vars_ = scores["arms"], scores["vars"]
        leads = list(arms[next(iter(arms))].get("leads", (24, 48, 72, 96, 120, 144)))
        iz, il, hl = vars_.index("z"), len(leads) - 1, leads[-1]  # the longest lead of the run
        show = [h for h in SHOW_H if h in leads] or [hl]
        idx = [leads.index(h) for h in show]
        show_s = " / ".join(f"{h}" for h in show) + "h"
        present = [(k, n) for k, n in ARMS if k in arms]
        guide_arms = [(k, n) for k, n in present if k != REF]
        boot_key = next((k for k in scores if k.startswith("bootstrap_vs_")), None)
        boot = scores.get(boot_key, {}) if boot_key else {}
        items = []
        if REF in arms and guide_arms:
            k0, n0 = guide_arms[0]
            rz, bz = arms[REF]["rmse"][il][iz], arms[k0]["rmse"][il][iz]
            rc, bc = arms[REF]["crps"][il][iz], arms[k0]["crps"][il][iz]
            items.append(f"{n0}: z500 {hl} h RMSE {bz:.0f} (회귀 앙상블 {rz:.0f}"
                         + (f", ClimODE 논문 {paper_z[hl]:.0f}" if hl in paper_z else "")
                         + f"), z500 {hl} h CRPS {bc:.4f} vs {rc:.4f} ({100 * (rc - bc) / rc:+.1f} %).")
            fair = {n: sum(sum(r) for r in arms[k]["crps"]) / (len(leads) * len(vars_)) for k, n in present}
            items.append("전 변수·전 lead 평균 CRPS (정규화): " + ", ".join(f"{n} {v:.4f}" for n, v in fair.items()) + ".")
            if boot:
                sig = {n: sum(h < 0 for row in boot[k]["crps"]["hi"] for h in row)
                       for k, n in guide_arms if k in boot}
                items.append(f"Bootstrap 95 % CI (vs {boot_key.replace('bootstrap_vs_', '')}): CRPS가 유의하게 낮은 "
                             f"(변수 × lead) 셀 수 / {len(leads) * len(vars_)}: "
                             + ", ".join(f"{n} {c}" for n, c in sig.items()) + ".")
            ssr = {n: rng_text([arms[k]["ssr"][i][iz] for i in range(len(leads))]) for k, n in present}
            items.append("z500 SSR (lead 범위): " + ", ".join(f"{n} {v}" for n, v in ssr.items())
                         + f". {hl} h z500 하위 tercile BSS: {n0} {arms[k0]['bss_lower'][il][iz]:.2f}.")
            pk = "power_ratio_k13-32"
            if pk in arms[k0]:
                items.append(f"작은 스케일 파워 (k13–32, 1이 완벽) z500 {hl} h: {n0} {arms[k0][pk][il][iz]:.1f}, "
                             f"회귀 {arms[REF][pk][il][iz]:.1f}.")
        if flowdep:
            corr = {k: rng_text([v["case_corr"] for v in flowdep[k].values()]) for k in flowdep}
            items.append("Flow-dependence (spread–|오차| case 상관, z500/t850 24–144 h): "
                         + ", ".join(f"{k.replace('seqfm_dct_pinn_', '')} {v}" for k, v in corr.items()) + ".")
        story += bullets(items or ["score json에 비교할 arm이 없음."], st)
    else:
        missing.append(scores_name)
        story += bullets([f"{scores_name} 없음: 결론을 채우지 못함."], st)

    # -- 3. base models, from the stage 2 test dicts -------------------------------------------------
    story.append(Paragraph("3. 기본 모델 (guide 이전, 테스트 기간, z500)", st["h1"]))
    first = (fm or reg or {}).get("test", {}).get("raw", {}).get("rmse", {"24": 0})
    h0, h1 = min(first, key=int), max(first, key=int)  # shortest and longest lead of the stage 2 run
    rows = [["모델", f"RMSE {h0} / {h1}h (m²/s²)", f"CRPS {h0} / {h1}h", f"SSR {h0} / {h1}h", "정역학 잔차"]]
    truth_hyd = None
    for r, name in ((fm, "seqfm_dct"), (reg, "seqregress")):
        if not r:
            continue
        t = r["test"]
        truth_hyd = t.get("truth_residual", {}).get("hyd", truth_hyd)
        for arm, suffix in (("raw", " raw"), ("pinn", " + PINN")):
            if arm in t:
                q = t[arm]
                rows.append([name + suffix, fmt3([q["rmse"][h0]["z"], q["rmse"][h1]["z"]], 0),
                             fmt3([q["crps_ens"][h0]["z"], q["crps_ens"][h1]["z"]], 4),
                             fmt3([q["ssr"][h0]["z"], q["ssr"][h1]["z"]], 2), f"{q['residual']['hyd']:.2f}"])
    if len(rows) > 1:
        story.append(table(rows, [38 * mm, 38 * mm, 32 * mm, 28 * mm, 26 * mm], st))
        story.append(Spacer(1, 3))
        items = ["CRPS는 앙상블 CRPS (정규화 z500). 잔차는 PINN guide의 방정식별 정규화 RMS"
                 + (f"; 실제 대기 (truth) 정역학 잔차 {truth_hyd:.2f}." if truth_hyd is not None else ".")]
        if fm and reg and "raw" in fm["test"] and "raw" in reg["test"]:
            items.append(f"FM 단독 z500 {h0} h RMSE {fm['test']['raw']['rmse'][h0]['z']:.0f} vs 회귀 "
                         f"{reg['test']['raw']['rmse'][h0]['z']:.0f}; FM SSR {h0} h {fm['test']['raw']['ssr'][h0]['z']:.2f}.")
        story += bullets(items, st)
    else:
        story += bullets(["단계 2 json 없음."], st)

    # -- 4. guide arms, from the score json ----------------------------------------------------------
    story.append(Paragraph("4. Guide arm vs 회귀 앙상블 (테스트 기간)", st["h1"]))
    if scores:
        rows = [["arm", f"z500 RMSE<br/>{show_s}", f"t850 RMSE (K)<br/>{show_s}", f"t2m RMSE (K)<br/>{show_s}",
                 f"z500 CRPS<br/>{show_s}", f"z500 SSR<br/>{show_s}"]]
        for key, name in present:
            a = arms[key]
            pick = lambda m, v: [a[m][i][vars_.index(v)] for i in idx]  # noqa: E731
            rows.append([name, fmt3(pick("rmse", "z"), 0), fmt3(pick("rmse", "t"), 2), fmt3(pick("rmse", "t2m"), 2),
                         fmt3(pick("crps", "z"), 4), fmt3(pick("ssr", "z"), 2)])
        story.append(table(rows, [24 * mm, 25 * mm, 27 * mm, 27 * mm, 38 * mm, 33 * mm], st,
                           bold_rows=tuple(i + 1 for i, (k, _) in enumerate(present) if k != REF)[:1]))
        story.append(Spacer(1, 3))
        items = ["CRPS는 z500을 정규화한 값. 같은 평균장을 공유하는 arm은 RMSE가 동일."]
        if boot:
            hi_max = {n: max(max(r) for r in boot[k]["crps"]["hi"]) for k, n in guide_arms if k in boot}
            items.append("Bootstrap CRPS 차이의 95 % CI 상한 최댓값: "
                         + ", ".join(f"{n} {v:.1e}" for n, v in hi_max.items())
                         + " (음수 = 전 변수·전 lead에서 유의한 개선).")
        a = arms[guide_arms[0][0]] if guide_arms else {}
        doob, fall = a.get("doob_ratio_member"), a.get("falling_steps_member")
        if isinstance(doob, list) and len(doob) > iz and not isinstance(doob[0], list):  # one value per variable
            items.append(f"{guide_arms[0][1]} z500 Doob ratio {doob[iz]:.2f}"
                         + (f", falling steps {fall[iz]:.0f}" if isinstance(fall, list) and len(fall) > iz else "")
                         + " (에너지의 lead 단조성).")
        story += bullets(items, st)
    else:
        story += bullets([f"{scores_name} 없음."], st)

    # -- 5. figures ----------------------------------------------------------------------------------
    story.append(Paragraph("5. 그림", st["h1"]))
    n_fig = 0
    for name, cap in (("fig_broad.png", "RMSE (위)와 CRPS / spread (아래): 회귀 앙상블, regm arm, persistence, "
                                        "ClimODE 논문 값 (plot_step13_broad.py)."),
                      ("fig_broad_vs_climode.png", "같은 테스트 IC에서의 결정론적 RMSE: ClimODE 논문, 4년 학습 step 13, "
                                                   "이번 broad 학습.")):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            n_fig += 1
            story += [figure(path, full_w), Paragraph(f"그림 {n_fig}. {cap}", st["cap"]), Spacer(1, 8)]
    tpath = training_figure(runs, out_dir, stages)
    if tpath:
        n_fig += 1
        story += [figure(tpath, full_w),
                  Paragraph(f"그림 {n_fig}. 학습 곡선 (<tag>.pt_metrics.csv): 학습 손실과 검증 손실, 선택된 iteration.",
                            st["cap"])]
    if not n_fig:
        story += bullets(["그림 없음 (plot_step13_broad.py를 먼저 실행)."], st)

    # -- 6. flow dependence --------------------------------------------------------------------------
    story.append(PageBreak())
    story.append(Paragraph("6. Flow-dependence (spread가 상황별 오차를 따라가는가)", st["h1"]))
    if flowdep:
        rows = [["arm", "spread CV", "spread–|err| 상관", "spread bin (하→상)", "RMSE bin (하→상)"]]
        for k, v in flowdep.items():
            vals = list(v.values())
            rows.append([k.replace("seqfm_dct_pinn_", ""), rng_text([q["cv_spread"] for q in vals]),
                         rng_text([q["case_corr"] for q in vals]),
                         f"{rng_text([q['bin_spread'][0] for q in vals])} → {rng_text([q['bin_spread'][-1] for q in vals])}",
                         f"{rng_text([q['bin_rmse'][0] for q in vals])} → {rng_text([q['bin_rmse'][-1] for q in vals])}"])
        story.append(table(rows, [36 * mm, 24 * mm, 34 * mm, 34 * mm, 34 * mm], st))
        story.append(Spacer(1, 3))
        keys = sorted({q for v in flowdep.values() for q in v})
        story += bullets([f"flowdep.json의 {', '.join(keys)} 전체에 대한 범위. spread를 5개 bin으로 나눈 뒤 bin별 "
                          "평균 spread와 RMSE (전체 평균으로 정규화)의 첫 bin → 마지막 bin."], st)
    else:
        missing.append("flowdep.json")
        story += bullets(["flowdep.json 없음."], st)

    # -- 7. operator ablation ------------------------------------------------------------------------
    story.append(Paragraph("7. Operator ablation (결정론적 평균, 테스트 기간)", st["h1"]))
    if operator:
        t = operator["test"]
        names = [k for k in ("mean", "mbmx", "operator", "operator_mean_only", "persistence") if k in t]
        hs = [h for h in SHOW_H if str(h) in t[names[0]]["rmse"]] or [max(t[names[0]]["rmse"], key=int)]
        hs_s = " / ".join(map(str, hs)) + "h"
        rows = [["평균장", f"z500 RMSE {hs_s}", f"t850 RMSE {hs_s} (K)"]]
        for k in names:
            rows.append([k + (" (seqfm_dct_pinn)" if k == "mean" else ""),
                         fmt3([t[k]["rmse"][str(h)]["z"] for h in hs], 0),
                         fmt3([t[k]["rmse"][str(h)]["t"] for h in hs], 2)])
        best = min((k for k in names if k != "persistence"), key=lambda k: t[k]["rmse"][str(hs[-1])]["z"])
        story.append(table(rows, [50 * mm, 60 * mm, 60 * mm], st, bold_rows=(names.index(best) + 1,)))
        story.append(Spacer(1, 3))
        items = [f"FM 앙상블 위의 평균장 중 {best}가 z500 {hs[-1]} h에서 최선"]
        if reg and "pinn" in reg["test"] and all(str(h) in reg["test"]["pinn"]["rmse"] for h in hs):
            q = reg["test"]["pinn"]["rmse"]
            items[0] += (f"; 회귀 평균 (z500 {fmt3([q[str(h)]['z'] for h in hs], 0)})이 단기에서 더 낮아 "
                         "guide arm의 평균장은 회귀 평균을 씀.")
        story += bullets(items, st)
    else:
        story += bullets(["operator_seqfm_dct_pinn.json 없음 (operator ablation을 돌리지 않음)."], st)

    # -- 8. run records ------------------------------------------------------------------------------
    story.append(Paragraph("8. 학습 기록", st["h1"]))
    story.append(table(run_rows(stages), [40 * mm, 24 * mm, 36 * mm, 26 * mm, 20 * mm, 28 * mm], st))
    story.append(Spacer(1, 3))
    story += bullets(["device / commit / best it은 pgcfm.runs로 학습한 run에만 기록됨 (그 전 run은 '기록 없음')."], st)

    story.append(Paragraph("9. 다음 단계 제안 (2026-10-27 run 기준)", st["h1"]))
    story += bullets(NEXT_STEPS, st)
    if missing:
        story += [Spacer(1, 6), Paragraph("빠진 입력: " + ", ".join(missing), st["cap"])]

    out = os.path.join(out_dir, "step13_broad_report.pdf")
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="Step 13 broad run 결과 요약",
                            author="physical_guide_flowmatching")
    doc.build(story)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs/step13_broad", help="run directory with the json outputs")
    p.add_argument("--scores", default="score_best.json", help="score json inside --runs")
    p.add_argument("--out", default="docs/step13_broad")
    a = p.parse_args(argv)
    print(build(a.runs, a.out, a.scores))


if __name__ == "__main__":
    main()
