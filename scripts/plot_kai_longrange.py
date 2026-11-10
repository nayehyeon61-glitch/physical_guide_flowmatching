"""Lead-time figure of the long-range arms (scripts/run_kai_tendency.py --score).

    python scripts/plot_kai_longrange.py runs/kaitend/score_La.json runs/kaitend/score_La_raw.npz out.png \
        [--title "..."] tag[=label[=colour[=linestyle]]] ...

Panels: normalized RMSE (65 channels), its change against KAI, anomaly correlation (65 channels), z500 ACC,
z500 / t850 / t2m RMSE, z500 prediction / truth power at k1-9. Persistence and climatology are dotted.
A tag that starts with ``blend`` is drawn dashed in the colour of the model that is blended with KAI, ``loop``
(the two models mixed every step) dash-dotted. Other combinations (seq_, cas_, three-way blends) take their
label, colour and line style from the command line.
"""

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# one colour per model, fixed (the same model keeps its colour in every figure)
COLOR = {"core_gn": "#2a78d6", "core_gn2": "#2a78d6",
         "hier_core_gn3+core_gn": "#eb6834",
         "hier_core_gn5+core_gn3+core_gn": "#1baf7a",
         "core_gnk8": "#eda100", "core_gnh": "#1baf7a", "core_gnm": "#6250d6", "core_gnmk8": "#e87ba4",
         "hier_core_gn5+core_gn3+kai": "#e87ba4", "hier_core_gn3+kai": "#e87ba4",
         "hier_core_gn5+core_gn3+core_gnk8": "#6250d6", "hier_core_gn5w+core_gn3+core_gn": "#6250d6"}
LABEL = {"kai": "KAI original", "kai_ro": "KAI + rollout fine-tuning",
         "core_gn": "daily core (1-day field, rollout 1-4 d)",
         "core_gn2": "daily core, 2 Heun steps per day",
         "core_gnk8": "daily core, rollout 1-8 d (learning extended)",
         "core_gnh": "daily core + previous day as input (memory of 1 day)",
         "core_gnm": "daily core + GRU memory (8 days before the IC)",
         "core_gnmk8": "daily core + GRU memory, rollout 1-8 d",
         "hier_core_gn3+core_gn": "3-day field + daily core",
         "hier_core_gn5+core_gn3+core_gn": "5-day + 3-day field + daily core",
         "hier_core_gn3+kai": "3-day field + KAI for single days",
         "hier_core_gn5+core_gn3+kai": "5-day + 3-day field + KAI for single days",
         "hier_core_gn5+core_gn3+core_gnk8": "5-day + 3-day field + daily core (1-8 d)",
         "hier_core_gn5w+core_gn3+core_gn": "wide 5-day + 3-day field + daily core"}


def style(tag, label=None, color=None, ls=None):
    st = _style(tag, label)
    if color:
        st["color"] = color
    if ls:
        st["ls"] = ls
    return st


def _style(tag, label=None):
    if tag == "persistence":
        return dict(color=MUTED, ls=":", lw=1.4, label="persistence")
    if tag == "climatology":
        return dict(color="#a5a49f", ls=":", lw=1.4, label="climatology")
    if tag in ("kai", "kai_ro"):
        return dict(color=INK, lw=2.4, ls="-" if tag == "kai" else "-.", label=label or LABEL[tag])
    base = tag.split("@")[0]
    blend = base.startswith("blend_") or base.startswith("loop_")
    model = base.split("_", 1)[1] if blend else base
    lab = label or (("KAI blended with: " if base.startswith("blend_") else "KAI mixed every step with: ")
                    + LABEL.get(model, model) if blend else LABEL.get(model, model))
    ls = "--" if base.startswith("blend_") else "-." if blend else "-"
    return dict(color=COLOR.get(model, "#e34948"), lw=2, ls=ls, label=lab)


def main(argv):
    title = "Long range: extending the tendency field in time (ERA5 5.625 deg daily means, test 2017)"
    if "--title" in argv:
        i = argv.index("--title")
        title = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    S = json.load(open(argv[0]))
    raw = np.load(argv[1])
    out = argv[2]
    tags = [(t.split("=") + [None] * 3)[:4] for t in argv[3:]]
    names, L, A = S["channels"], S["leads"], S["arms"]
    days = np.arange(1, L + 1)
    sc = [i for i, n in enumerate(names) if n not in ("orog", "tisr")]
    ch = names.index

    def acc_mean(k):
        return np.array(A[k]["acc"])[:, sc].mean(1)

    def power(k, c, lo, hi):
        return raw[f"{k}__sp_pred"][:, c, lo:hi + 1].sum(-1) / raw[f"{k}__sp_true"][:, c, lo:hi + 1].sum(-1)

    kai = np.array(A["kai"]["nrmse"])
    panels = [
        ("normalized RMSE, 65 channels", lambda k: np.array(A[k]["nrmse"]), True, None),
        ("normalized RMSE against KAI (%)", lambda k: 100 * (np.array(A[k]["nrmse"]) / kai - 1), False, 0.0),
        ("anomaly correlation, 65 channels", acc_mean, True, None),
        ("z500 anomaly correlation", lambda k: np.array(A[k]["acc"])[:, ch("z500")], True, None),
        ("z500 RMSE (m$^2$/s$^2$)", lambda k: np.array(A[k]["rmse"])[:, ch("z500")], True, None),
        ("t850 RMSE (K)", lambda k: np.array(A[k]["rmse"])[:, ch("t850")], True, None),
        ("t2m RMSE (K)", lambda k: np.array(A[k]["rmse"])[:, ch("t2m")], True, None),
        ("z500 forecast / truth power, k1-9 (1 = full amplitude)", lambda k: power(k, ch("z500"), 1, 9), False, 1.0),
    ]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK})
    fig, ax = plt.subplots(2, 4, figsize=(19, 8.6), facecolor=SURFACE)
    handles = {}
    for p, (ttl, fn, refs, hline) in zip(ax.flat, panels):
        p.set_facecolor(SURFACE)
        if refs:
            for k in ("persistence", "climatology"):
                if k == "climatology" and "correlation" in ttl:
                    continue
                st = style(k)
                handles[st["label"]] = p.plot(days, fn(k), **st)[0]
        for tag, lab, col, ls in tags:
            if tag not in A or (tag == "kai" and "against KAI" in ttl):
                continue
            st = style(tag, lab, col, ls)
            handles[st["label"]] = p.plot(days, fn(tag), **st)[0]
        if hline is not None:
            p.axhline(hline, color=INK, lw=0.8)
        if ttl.endswith("65 channels") and "RMSE" in ttl:
            p.set_ylim(0.15, 0.7)
        if "RMSE (" in ttl:  # a diverging line leaves the panel instead of flattening the others
            p.set_ylim(0, 1.15 * float(fn("persistence").max()))
        p.set_title(ttl, fontsize=10.5, loc="left")
        p.set_xlabel("lead (days)")
        p.set_xticks(days)
        p.grid(color=GRID, lw=0.8)
        for s in ("top", "right"):
            p.spines[s].set_visible(False)
    fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=3, fontsize=9.5, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"{title}, {S['ics']} ICs", fontsize=12.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.035 + 0.028 * ((len(handles) + 2) // 3), 1, 0.96))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)
    for tag, *_ in tags:  # the table behind the figure
        if tag in A:
            print(f"{tag:42s} nrmse", "/".join(f"{A[tag]['nrmse'][q]:.3f}" for q in (0, 2, 4, 6, L - 1)),
                  " acc", "/".join(f"{acc_mean(tag)[q]:.3f}" for q in (0, 2, 4, 6, L - 1)),
                  " z500 k1-9 power", "/".join(f"{power(tag, ch('z500'), 1, 9)[q]:.2f}" for q in (0, 4, L - 1)))


if __name__ == "__main__":
    main(sys.argv[1:])
