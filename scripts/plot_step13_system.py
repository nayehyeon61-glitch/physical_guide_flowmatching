"""Step 13 training system diagram (docs/step13/fig_system.png): the three stages as built in
run_step13_pinn.py, run_step13_seqfm.py, run_step13_guide.py and run_step13_operator.py.
``--best`` draws fig_system_best.png: the same system with the parts the best ensemble (regm_band:
a + b * regression mean + c * x(t), FM + PINN deviations, closed-form band Gamma) does not use faded.
``--sigma`` draws fig_system_sigma.png: how the flow-dependent spread field of regm_flowop is trained
(docs/step13_sequence.md 4.3, pgcfm/flow_sigma.py: OperatorSigma).

    python scripts/plot_step13_system.py --out docs/step13 [--best | --sigma]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

KIND = {  # fill, edge
    "data": ("#f4f5f8", "#6b7894"),
    "op": ("#ebf2fc", "#2e62a6"),
    "net": ("#eaf5f1", "#1f8a70"),
    "loss": ("#fcf2e8", "#c0692a"),
    "energy": ("#f2ecf8", "#7d5ba8"),
}
STAGE = {1: "#1f8a70", 2: "#2e62a6", 3: "#7d5ba8"}
TITLE, BODY, RULE = "#1e2b45", "#5a6681", "#d9dce3"
W, H = 15.5, 8.5
COLS = [4 + i * (W + 3.75) for i in range(5)]
FADE = {"on": False}  # --best: alpha of the parts the best ensemble does not use


def alpha(dim):
    return 0.2 if dim and FADE["on"] else 1.0


def cx(c):
    return COLS[c] + W / 2


def box(ax, c, y, title, lines=(), kind="data", frozen=False, w=W, x=None, dim=False):
    x = COLS[c] if x is None else x
    fill, edge = KIND[kind]
    al = alpha(dim)
    ax.add_patch(FancyBboxPatch((x, y - H / 2), w, H, boxstyle="round,pad=0,rounding_size=0.9", fc=fill, ec=edge,
                                lw=1.6, ls=(0, (4, 2.5)) if frozen else "-", alpha=al))
    n = len(lines)
    top = y + (0.35 + 0.9 * n) if n else y
    ax.text(x + w / 2, top, title, ha="center", va="center", fontsize=13.5, weight="bold", color=TITLE, alpha=al)
    for i, s in enumerate(lines):
        ax.text(x + w / 2, top - 2.35 - 1.75 * i, s, ha="center", va="center", fontsize=10.8, color=BODY, alpha=al)


def arrow(ax, pts, color, ls="-", label=None, lpos=None, halo=False, dim=False):
    """Polyline through pts with an arrow head on the last segment."""
    kw = dict(color=color, lw=1.6, ls=ls, alpha=alpha(dim))
    pe_ = [pe.withStroke(linewidth=6, foreground="white")] if halo else None
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], solid_capstyle="butt", path_effects=pe_, **kw)
    a = FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=15, shrinkA=0, shrinkB=0,
                        color=color, lw=1.6, ls=ls, alpha=alpha(dim))
    if halo:
        a.set_path_effects(pe_)
    ax.add_patch(a)
    if label:
        ax.text(*lpos, label, fontsize=10.5, color=color, ha="left", va="center", alpha=alpha(dim))


def right(c, y):
    return (COLS[c] + W + 0.25, y)


def left(c, y):
    return (COLS[c] - 0.35, y)


def row(ax, c0, c1, y, color, dim=False):
    arrow(ax, [right(c0, y), left(c1, y)], color, dim=dim)


def sublabel(ax, x, y, text, color, dim=False):
    ax.text(x, y, text, fontsize=12, color=color, weight="bold", ha="left", va="center", alpha=alpha(dim))


def stage_band(ax, y0, y1, n, name, color):
    ax.plot([2.4, 2.4], [y0, y1], color=color, lw=3, solid_capstyle="round")
    ax.text(1.1, (y0 + y1) / 2, f"0{n}   {name}", rotation=90, ha="center", va="center", fontsize=13.5,
            weight="bold", color=color)


def sigma_system(out):
    """regm_flowop: the owner's operator pretrained on the energy-refined members, fine-tuned into sigma(x)."""
    fig = plt.figure(figsize=(20, 14.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(6.5, 77)
    ax.axis("off")
    ax.text(4, 73.6, "Flow-dependent spread field", fontsize=30, weight="bold", color=TITLE, va="center")
    ax.text(4, 70.3, "regm_flowop: the operator learns the main stream on the energy-refined ensemble, then a new head "
            "turns its features into σ(x)", fontsize=15, color=BODY, va="center")
    ax.text(99, 73.6, "Fitted on 2016 (every 5th IC held out)", fontsize=12.5, color=BODY, ha="right", va="center")
    ax.text(99, 71.2, "Stages 1-2 (PINN, FM, regression) and band Γ frozen", fontsize=11.5, color=BODY, ha="right",
            va="center")
    ax.text(99, 69.0, "Test 2017–2018 (ClimODE's 360 ICs), 8 members", fontsize=11.5, color=BODY, ha="right",
            va="center")
    ax.plot([4, 99], [67.2, 67.2], color=RULE, lw=1.2)

    g = STAGE[2]
    stage_band(ax, 53.3, 65.2, 1, "PRETRAIN", g)
    sublabel(ax, 4, 64.6, "Step 1 · operator: energy-refined ensemble → deterministic forecast", g)
    y = 58
    box(ax, 0, y, "regm_band members", ["μ* + IDCT(Γ_b · DCT(x_m − x̄))", "8 per IC, 2016", "weights frozen"], "energy",
        frozen=True)
    box(ax, 1, y, "Member encoder", ["FNO on [member, member − mean]", "mean pooling over 8", "→ spread structure"],
        "net")
    box(ax, 2, y, "Global FNO", ["[ensemble mean, pooled code,", "x(t), lead, time of year / day]",
                                 "width 32, 8 modes"], "net")
    box(ax, 3, y, "Old head", ["mean + 1×1 conv (zero init)", "deterministic field per lead"], "op")
    box(ax, 4, y, "MSE to the truth", ["latitude weighted, 40 epochs", "checkpoint on held-out ICs"], "loss")
    for c in range(4):
        row(ax, c, c + 1, y, g)

    o = STAGE[3]
    stage_band(ax, 37.5, 49.6, 2, "FINE-TUNE", o)
    yl = 50.6
    arrow(ax, [(cx(2), 58 - H / 2 - 0.2), (cx(2), yl), (cx(0), yl), (cx(0), 42 + H / 2 + 0.35)], g,
          ls=(0, (4, 2.5)), label="pretrained weights", lpos=(cx(0) + 0.8, 48.6))
    sublabel(ax, cx(0) + 13.5, 48.6, "Step 2 · new head: features → σ(x), the old head is dropped", o)
    y = 42
    box(ax, 0, y, "Operator features f", ["global FNO output", "before the old head", "frozen in a), tuned in b)"],
        "net", frozen=True)
    box(ax, 1, y, "New σ head", ["1×1 conv, zero init → g", "σ = σ_clim · exp(g), |g| ≤ 3",
                                 "starts at the climatology"], "net")
    box(ax, 2, y, "Gaussian NLL", ["log σ + (O − μ*)² / (2σ²)", "ClimODE's utils.nll", "no spread label needed"],
        "loss")
    box(ax, 3, y, "Two training stages", ["a) operator frozen: head, 30 ep", "b) everything, lr 1e-4, 10 ep",
                                          "best held-out 2016 NLL"], "op")
    box(ax, 4, y, "Checks", ["corr(σ, |error|) per grid point", "factor CV over cases", "rank histogram, CRPS"],
        "data")
    for c in range(4):
        row(ax, c, c + 1, y, o)

    v = STAGE[1]
    stage_band(ax, 21.8, 34.0, 3, "APPLY", v)
    sublabel(ax, COLS[2], 33.4, "Step 3 · rescale the members case by case, mean and average spread unchanged", v)
    y = 27
    box(ax, 0, y, "regm_band members", ["y_m, 2017–2018", "same members as the", "reported model"], "energy",
        frozen=True)
    box(ax, 1, y, "σ(x) per case", ["operator + new head", "per lead · variable", "· grid point"], "net",
        frozen=True)
    box(ax, 2, y, "Factor f = σ / ρ", ["ρ² = 2016 mean of σ²", "so E₂₀₁₆ f² = 1"], "energy")
    box(ax, 3, y, "μ* + f ⊙ (y_m − μ*)", ["mean μ* unchanged (same RMSE)", "band Γ spread kept on average",
                                         "f moves it between cases"], "energy")
    box(ax, 4, y, "regm_flowop", ["8 members, 6 days", "spread follows the flow"], "data")
    for c in range(4):
        row(ax, c, c + 1, y, v)
    arrow(ax, [(cx(1), 42 - H / 2 - 0.2), (cx(1), 27 + H / 2 + 0.35)], o, ls=(0, (4, 2.5)), label="trained head",
          lpos=(cx(1) + 0.8, 33.4), halo=True)

    fill, edge = KIND["data"]
    ax.add_patch(FancyBboxPatch((4, 11.6), 95, 6.6, boxstyle="round,pad=0,rounding_size=0.9", fc=fill, ec=edge, lw=1.6))
    ax.text(51.5, 16.2, "Option 1 for comparison (regm_flow)", ha="center", va="center", fontsize=14, weight="bold",
            color=TITLE)
    ax.text(51.5, 13.4, "the same σ = σ_clim · exp(g), NLL, factor and members, but g from ClimODE's uncertainty ResNet "
            "on hand-made features [x(t), μ*, μ* − x(t), lead, time, static]", ha="center", va="center", fontsize=11.5,
            color=BODY)
    ax.plot([4, 99], [10.3, 10.3], color=RULE, lw=1.2)
    ax.text(4, 9.0, "Solid: data / loss flow.  Dashed border: weights frozen from an earlier step.  "
            "Orange: training losses.  Purple: energy control.", fontsize=11.5, color=BODY, va="center")
    ax.text(4, 7.4, "The NLL is smallest when σ matches the size of the actual error in each case, so the truth "
            "error of μ* teaches the spread; FM, PINN, μ* and Γ_b stay fixed.", fontsize=11.5, color=BODY, va="center")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "fig_system_sigma.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"[system] {path}")
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/step13")
    p.add_argument("--best", action="store_true", help="fade what the best ensemble (regm_band) does not use")
    p.add_argument("--sigma", action="store_true", help="the flow-dependent spread field (regm_flowop) instead")
    a = p.parse_args(argv)
    if a.sigma:
        return sigma_system(a.out)
    best = FADE["on"] = a.best
    fig = plt.figure(figsize=(20, 22.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.5, 100.5)
    ax.set_ylim(-0.5, 112)
    ax.axis("off")

    ax.text(4, 108.6, "Sequence PG-CFM", fontsize=30, weight="bold", color=TITLE, va="center")
    ax.text(4, 105.3, "Best ensemble (regm_band): a + b·μ_reg + c·x(t) + FM + PINN deviations + closed-form band Γ.  "
            "Faded: parts of the system it does not use" if best else
            "Step 13 training system: PINN pretraining → sequence flow matching → calibration and "
            "operator on frozen models", fontsize=15, color=BODY, va="center")
    ax.text(99, 108.6, "ERA5 5.625°, 66 KAI channels every 6 h", fontsize=12.5, color=BODY, ha="right", va="center")
    ax.text(99, 106.2, "Lossless 3-D DCT  ·  K = 24 steps (6 h to 144 h) in one flow", fontsize=11.5, color=BODY,
            ha="right", va="center")
    ax.text(99, 104.0, "Train 2012–2015  ·  fit stage 3 on 2016  ·  test 2017–2018", fontsize=11.5, color=BODY,
            ha="right", va="center")
    ax.plot([4, 99], [102.2, 102.2], color=RULE, lw=1.2)

    # -- stage 1: PINN pretraining ---------------------------------------------------------------------
    g = STAGE[1]
    stage_band(ax, 88.5, 100.2, 1, "PINN", g)
    sublabel(ax, 4, 99.6, "Stage 1 · pretrain the PINN projector, then freeze it", g)
    y = 93
    box(ax, 0, y, "Truth sequence", ["x(t),  d_k = x(t + 6k h) − x(t)", "k = 1 … 24, 66 channels"], "data")
    box(ax, 1, y, "Perturbed input d′", ["d + α·ε,  ε ~ per-lead DCT std", "stand-in for FM members"], "op")
    box(ax, 2, y, "PINN projector P", ["d′ + lowpass N(x(t), d′)", "corrects low 8×16 modes only", "identity skip"],
        "net")
    box(ax, 3, y, "Projection loss", ["|P(d′) − d′|² + λ_P · Φ(P(d′))", "Φ: 6 h primitive-equation", "residual"],
        "loss")
    box(ax, 4, y, "Gates on 2016", ["identity on the truth", "spread ratio through P ≈ 1", "Φ falls (hydrostatic)"],
        "data")
    for c in range(4):
        row(ax, c, c + 1, y, g)

    # -- stage 2: sequence models ----------------------------------------------------------------------
    b = STAGE[2]
    stage_band(ax, 46, 85.6, 2, "SEQUENCE MODELS", b)
    sublabel(ax, 4, 85, "Stage 2 · conditional flow matching on the whole 6-day sequence", b)
    y = 79
    box(ax, 0, y, "History + target", ["x(t − 12 h), x(t − 6 h), x(t)", "+ static fields, hour", "target d_1 … d_24"],
        "data")
    box(ax, 1, y, "3-D DCT of d", ["time × lat × lon, lossless", "standardized per lead", "and channel"], "op")
    box(ax, 2, y, "Conditional FM  v_θ", ["U-Net computed on the grid", "velocity in DCT coefficients"], "net")
    box(ax, 3, y, "Flow matching loss", ["|v_θ(a_s, s) − (a₁ − a₀)|²", "torchcfm, σ = 0"], "loss")
    for c in range(3):
        row(ax, c, c + 1, y, b)

    sublabel(ax, 4, 71.4, "Generation · v_θ and P frozen", b)
    y = 65.5
    box(ax, 0, y, "Gaussian noise a₀", ["8 members per IC", "same noise for every arm"], "data")
    box(ax, 1, y, "Flow ODE in s", ["Heun, 8 steps", "no guidance (raw flow)"] if best else
        ["Heun, 8 steps", "options: physics guidance (H3)", "or P inside the ODE"], "net", frozen=True)
    box(ax, 2, y, "Frozen PINN P", ["on every member (H2)", "in DCT space, low modes"], "net", frozen=True)
    box(ax, 3, y, "IDCT + x(t)", ["physical members x_m", "clamp q ≥ 0, tcwv ≥ 0"], "op")
    box(ax, 4, y, "FM + PINN ensemble", ["8 members, 6 days", "too much spread (SSR 1.6)"], "data")
    for c in range(4):
        row(ax, c, c + 1, y, b)

    sublabel(ax, 4, 57.9, "H1 control · regression with the same U-Net", b)
    y = 52
    box(ax, 0, y, "Same inputs", ["x(t − 12 h), x(t − 6 h), x(t)", "target d_1 … d_24"], "data")
    box(ax, 1, y, "Same U-Net, MSE", ["mean sequence, 144 h", "20 min on CPU"], "net")
    box(ax, 2, y, "Statistical errors", ["Gaussian, diagonal in the", "3-D DCT of 2016 residuals"], "op")
    box(ax, 3, y, "Regression ensemble", ["μ_reg + errors, then frozen P", "member mean → main stream"], "data")
    for c in range(3):
        row(ax, c, c + 1, y, b)

    # -- stage 3: calibration and operator -------------------------------------------------------------
    v = STAGE[3]
    stage_band(ax, 13.2, 41.6, 3, "CALIBRATION", v)
    y = 33.5
    yl = 44.0
    arrow(ax, [(cx(4), 65.5 - H / 2 - 0.2), (cx(4), yl), (cx(0), yl), (cx(0), y + H / 2 + 0.35)], v)
    sublabel(ax, cx(0) + 2, 40.8, "Stage 3 · energy guide, fitted on 2016 with every generator frozen", v)
    box(ax, 0, y, "Member deviations", ["x_m − x̄ of the FM + PINN", "members, 2-D DCT", "5 radial bands"], "op")
    box(ax, 1, y, "Band energy Γ_b", ["√(err² / ((1 + 1/M) spread²))", "per lead · variable · band",
                                      "err of μ* on 2016" if best else "or learned guide (fair CRPS)"],
        "energy")
    box(ax, 2, y, "Scaled deviations", ["IDCT(Γ_b · DCT(x_m − x̄))", "energy ≤ FM energy where Γ ≤ 1"], "energy")
    box(ax, 3, y, "+ main stream μ*", ["μ* = a + b·μ_reg + c·x(t)", "a, b, c per lead · variable", "fitted on 2016"] if best else
        ["a + b·μ_reg + c·x(t) (best)", "or μ_reg, operator, step 12 core"], "data")
    box(ax, 4, y, "Calibrated ensemble", ["y_m = μ* + scaled deviation", "Doob check on the energy",
                                          "8 members, 6 days"], "energy")
    for c in range(4):
        row(ax, c, c + 1, y, v)
    arrow(ax, [(cx(3), 52 - H / 2 - 0.2), (cx(3), y + H / 2 + 0.35)], b, label="μ_reg", lpos=(cx(3) + 0.8, 40.8),
          halo=True)

    sublabel(ax, 4, 25.3, "Operator path · the owner's stage 3" + ("  (not used by the best ensemble)" if best else ""),
             v, dim=True)
    y = 19.5
    box(ax, 1, y, "2016 ensembles", ["frozen FM + PINN members", "or regression members"], "data", dim=True)
    box(ax, 2, y, "Neural operator", ["EnsembleToDeterministic FNO", "member encoder + pooling", "zero head = mean"],
        "net", dim=True)
    box(ax, 3, y, "Deterministic forecast", ["one field per lead", "alternative μ*"], "data", dim=True)
    box(ax, 4, y, "Operator loss", ["latitude-weighted MSE", "to the truth"], "loss", dim=True)
    row(ax, 1, 2, y, v, dim=True)
    row(ax, 2, 3, y, v, dim=True)
    row(ax, 3, 4, y, v, dim=True)
    arrow(ax, [(cx(3), y + H / 2 + 0.2), (cx(3), 33.5 - H / 2 - 0.35)], v, ls=(0, (4, 2.5)), label="alt. μ*",
          lpos=(cx(3) + 0.8, 26.6), dim=True)

    # -- inference summary -----------------------------------------------------------------------------
    fill, edge = KIND["data"]
    ax.add_patch(FancyBboxPatch((4, 4.6), 95, 6.6, boxstyle="round,pad=0,rounding_size=0.9", fc=fill, ec=edge, lw=1.6))
    ax.text(51.5, 9.2, "Inference, 2017–2018 (ClimODE's 360 ICs)", ha="center", va="center", fontsize=14,
            weight="bold", color=TITLE)
    ax.text(51.5, 6.4, "history → regression + errors → P → μ_reg → a + b·μ_reg + c·x(t) = μ*     ·     noise → flow ODE → P → "
            "IDCT → deviations → Γ_b → + μ* → 8 members" if best else
            "history → regression → MBM → μ*     ·     noise → flow ODE → P → IDCT → deviations → Γ_b → + μ* "
            "→ 8 members     ·     ensemble → operator → deterministic", ha="center", va="center", fontsize=11.5,
            color=BODY)
    ax.plot([4, 99], [3.3, 3.3], color=RULE, lw=1.2)
    ax.text(4, 2.0, "Solid: data / loss flow.  Dashed border: weights frozen from an earlier stage.  "
            "Orange: training losses.  Purple: energy control.", fontsize=11.5, color=BODY, va="center")
    ax.text(4, 0.4, "The FM never trains through P or Γ. Current best: a + b·regression mean + c·x(t), FM deviations, "
            "closed-form band Γ (regm_band); the learned guide moves the energy to the largest band.", fontsize=11.5, color=BODY,
            va="center")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "fig_system_best.png" if best else "fig_system.png")
    fig.savefig(path, dpi=100, facecolor="white")
    plt.close(fig)
    print(f"[system] {path}")
    return path


if __name__ == "__main__":
    main()
