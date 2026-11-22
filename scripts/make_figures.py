#!/usr/bin/env python
"""Draw the report figures from results/summary.json into results/figures/*.png.

Style: light chart surface, hairline solid grid, thin marks (2 px lines, 8 px markers, 2 px surface gap
between bars), categorical colours in a fixed validated order, blue/red poles for signed values,
a one-hue blue ramp for magnitudes, legends for every multi-series chart plus selective direct labels.
Every figure has a table twin in results/*.md / *.csv.

Usage:  python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS, NEG = "#2a78d6", "#e34948"                     # diverging poles (blue / red)
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PRETTY = {"detailed": "Detailed queries", "vague": "Vague queries (primary)", "vague+noise": "Vague + injected false memories"}
SHORT = {
    "BM25F (baseline)": "BM25F", "TF-IDF cosine (VSM)": "TF-IDF cosine", "Dense bi-encoder (bge-small, MaxP)": "Dense (bge-small)",
    "BM25 + dense (RRF)": "BM25 + dense (RRF)", "Clause-weighted BM25 (learned c_h, c_conf, c_ctx)": "Clause-weighted BM25*",
    "+ scope-aware negation (learned mu)": "+ scope-aware negation*", "+ Hinglish spelling handling": "+ Hinglish handling*",
    "Full LTR: coordinate ascent (ours)": "Full LTR, coord. ascent*", "Full LTR: pairwise logistic / RankNet-linear (ours)": "Full LTR, RankNet-linear*",
    "Full LTR: LambdaMART (ours)": "Full LTR, LambdaMART*",
}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "font.size": 10.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 12,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "lines.linewidth": 2, "lines.markersize": 8, "legend.frameon": False, "legend.fontsize": 9.5,
})


def short(name: str) -> str:
    return SHORT.get(name, name)


def finish(fig, name: str, note: str | None = None):
    if note:                                   # below the x-axis label; bbox_inches="tight" keeps it in frame
        fig.text(0.01, -0.035, note, color=MUTED, fontsize=8.5, ha="left", va="top")
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name)


def hgrid(ax):
    ax.grid(axis="x")
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- figures
def fig_main(S):
    rows = S["main"]["vague"]
    rows = sorted(rows, key=lambda r: r["rr"])
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    y = np.arange(len(rows))
    vals = [r["rr"] for r in rows]
    ax.barh(y, vals, height=0.68, color=SLOTS[0], edgecolor=SURF, linewidth=2)
    for i, r in enumerate(rows):
        ax.plot([r["mrr_ci_low"], r["mrr_ci_high"]], [i, i], color=INK2, linewidth=1)
    ax.set_yticks(y, [short(r["system"]) for r in rows])
    best = int(np.argmax(vals))
    base = next(i for i, r in enumerate(rows) if r["system"] == "BM25F (baseline)")
    for i in {best, base}:
        ax.text(rows[i]["mrr_ci_high"] + 0.01, i, f"{vals[i]:.3f}", va="center", color=INK, fontsize=9.5, fontweight="semibold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean reciprocal rank (MRR), 5-fold cross-validation")
    ax.set_title(f"Vague tip-of-the-tongue queries ({S['conditions']['vague']} queries, {S['n_films']} films)")
    hgrid(ax)
    finish(fig, "fig1_main_results.png", "* = our components. Whiskers: 95% bootstrap confidence interval.")


def fig_conditions(S):
    systems = [s for s in S["overview"]]
    conds = [c for c in ("detailed", "vague", "vague+noise") if c in S["main"]]
    order = sorted(systems, key=lambda s: S["overview"][s]["vague"]["mrr"])
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for i, s in enumerate(order):
        v = [S["overview"][s][c]["mrr"] for c in conds]
        ax.plot([min(v), max(v)], [i, i], color=GRID, linewidth=2, zorder=1)
        for j, c in enumerate(conds):
            ax.scatter(S["overview"][s][c]["mrr"], i, s=64, color=SLOTS[j], edgecolor=SURF, linewidth=2, zorder=3,
                       label=PRETTY[c] if i == 0 else None)
    ax.set_yticks(range(len(order)), [short(s) for s in order])
    ax.set_xlabel("MRR")
    ax.set_xlim(0.3, 1.0)
    ax.set_title("Every system on all three query conditions", pad=30)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2, handletextpad=0.3)
    hgrid(ax)
    finish(fig, "fig2_conditions.png", "* = our components.")


def fig_success(S):
    curves = S["success_curves"]["vague"]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ks = np.arange(1, 51)
    names = list(curves)
    for j, n in enumerate(names):
        ax.plot(ks, curves[n], color=SLOTS[j], label=short(n))
    for j, n in enumerate(names):
        if n in ("BM25F (baseline)", "Full LTR: coordinate ascent (ours)", "Dense bi-encoder (bge-small, MaxP)"):
            ax.annotate(short(n), (ks[-1], curves[n][-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 5, 10, 20, 50], ["1", "2", "3", "5", "10", "20", "50"])
    ax.set_xlim(1, 50)
    ax.set_ylim(0.3, 1.01)
    ax.set_xlabel("k (log scale)")
    ax.set_ylabel("Success@k  (= Recall@k)")
    ax.set_title("How often the right film is in the top k (vague queries)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=8.8)
    finish(fig, "fig3_success_at_k.png", "* = our components.")


def fig_ablation(S):
    conds = [c for c in ("vague", "vague+noise") if c in S.get("ablation", {})]
    fig, axes = plt.subplots(1, len(conds), figsize=(9.6, 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        rows = [r for r in S["ablation"][c] if r["variant"] != "Full model"]
        full = next(r for r in S["ablation"][c] if r["variant"] == "Full model")["mrr"]
        labels = [r["variant"].replace("– ", "without ").replace(" (title/person/year/language/genre)", "") for r in rows]
        d = [r["delta_mrr"] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, d, height=0.66, color=[NEG if v < 0 else POS for v in d], edgecolor=SURF, linewidth=2)
        ax.axvline(0, color=BASE, linewidth=1)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        lim = max(0.02, max(abs(v) for v in d) * 1.6)
        ax.set_xlim(-lim, lim)
        for i, v in enumerate(d):
            ax.text(v + (0.002 if v >= 0 else -0.002), i, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right",
                    fontsize=8.8, color=INK2)
        ax.set_title(f"{PRETTY[c]}\nfull model MRR = {full:.3f}", fontsize=11)
        ax.set_xlabel("Change in MRR when the component is removed")
        hgrid(ax)
    finish(fig, "fig4_ablation.png", "Red: removing the component lowers MRR (it helps). Blue: removing it raises MRR.")


def fig_sweeps(S):
    sw = S.get("hedge_sweep", {})
    conds = [c for c in ("vague", "vague+noise") if c in sw]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    for ax, param, title, xl in ((axes[0], "hedged", "Hedge weight", "c_h  (weight of hedged clauses)"),
                                 (axes[1], "mu", "Exclusion penalty", "μ  (weight of the exclusion penalty)")):
        for j, c in enumerate(conds):
            rows = [r for r in sw[c] if r["param"] == param]
            x = [r["value"] for r in rows]
            yv = [r["mrr_all"] for r in rows]
            ax.plot(x, yv, color=SLOTS[j], marker="o", markersize=6, markeredgecolor=SURF, label=PRETTY[c])
            learned = S["learned_weights"][c]["+ Hinglish spelling handling"]["all"][param]
            ax.axvline(learned, color=SLOTS[j], linewidth=1, alpha=0.6)
        ax.set_xlabel(xl)
        ax.set_ylabel("MRR (clause-weighted lexical ranker)")
        ax.set_title(title)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
    axes[0].legend(loc="center right", fontsize=8.8)
    fig.tight_layout()
    finish(fig, "fig5_weight_sweeps.png", "Vertical lines mark the value chosen by grid search on all queries of that condition.")


def fig_phenomena(S):
    cond = "vague+noise" if "vague+noise" in S.get("phenomena", {}) else "vague"
    rows = S["phenomena"][cond]
    systems = ["BM25F (baseline)", "Dense bi-encoder (bge-small, MaxP)", "Full LTR: coordinate ascent (ours)"]
    labels = [f"{r['phenomenon'].replace('_', ' ')}\n(n={r['n']})" for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for j, s in enumerate(systems):
        ax.bar(x + (j - 1) * w, [r[s] for r in rows], width=w, color=SLOTS[j], edgecolor=SURF, linewidth=2, label=short(s))
    ax.set_xticks(x, labels, fontsize=8.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("MRR")
    ax.set_title(f"MRR by tip-of-the-tongue phenomenon ({PRETTY[cond].lower()})", pad=30)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    finish(fig, "fig6_phenomena.png", "* = our components. Tags come from the query annotations; a query can carry several.")


def fig_negation(S):
    nm = S.get("negation_modes", {})
    conds = [c for c in ("vague", "detailed") if c in nm]
    modes = ["none", "drop", "penalize_all", "scope"]
    labels = ["none\n(bag of words)", "drop\nnegated words", "penalise\nevery negation", "scope-aware\nexclusions (ours)"]
    x = np.arange(len(modes))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    for j, c in enumerate(conds):
        vals = [next(r for r in nm[c] if r["mode"] == m)["mrr_negation_queries"] for m in modes]
        n = nm[c][0]["n_negation_queries"]
        ax.bar(x + (j - 0.5) * w, vals, width=w, color=SLOTS[j], edgecolor=SURF, linewidth=2, label=f"{PRETTY[c]} (n={n})")
        ax.text(x[-1] + (j - 0.5) * w, vals[-1] + 0.01, f"{vals[-1]:.3f}", ha="center", fontsize=9, color=INK, fontweight="semibold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("MRR on queries that rule something out")
    ax.set_title("How negations are handled", pad=30)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
    finish(fig, "fig7_negation_modes.png", "Clause weights re-learned by cross-validation for every mode.")


def fig_confusion(S):
    nb = S.get("nb")
    if not nb:
        return
    lab = nb["decade_confusion"]["labels"]
    M = np.array(nb["decade_confusion"]["matrix"], dtype=float)
    R = M / np.maximum(M.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("blue", [SURF] + BLUES)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.imshow(R, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(lab)):
        for j in range(len(lab)):
            if M[i, j]:
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8.5,
                        color="#ffffff" if R[i, j] > 0.55 else INK)
    ticks = [f"{d}s" for d in lab]
    ax.set_xticks(range(len(lab)), ticks, rotation=45)
    ax.set_yticks(range(len(lab)), ticks)
    ax.set_xlabel("Predicted decade")
    ax.set_ylabel("True decade")
    for s in ax.spines.values():
        s.set_visible(False)
    acc = nb["decade_docs_cv"]["accuracy"]
    ax.set_title(f"Naive Bayes decade prediction from plot text\naccuracy {acc:.2f} (majority baseline "
                 f"{nb['decade_docs_majority_baseline']['accuracy']:.2f}), 5-fold CV", fontsize=11)
    finish(fig, "fig8_nb_decade_confusion.png", "Cell shade = share of the true decade's films (row-normalised); numbers = film counts.")


def fig_weights(S):
    fm = S.get("final_model")
    if not fm:
        return
    items = sorted(fm["weights"].items(), key=lambda kv: kv[1])
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y = np.arange(len(items))
    ax.barh(y, vals, height=0.66, color=[POS if v >= 0 else NEG for v in vals], edgecolor=SURF, linewidth=2)
    ax.axvline(0, color=BASE, linewidth=1)
    ax.set_yticks(y, labels, fontsize=9)
    ax.set_xlabel("Weight on the standardised feature (weights sum to 1 in absolute value)")
    ax.set_title("What the learned ranker relies on")
    hgrid(ax)
    finish(fig, "fig9_feature_weights.png", f"Coordinate-ascent model trained on {fm['trained_on']} queries. Blue: raises the score; red: lowers it.")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 46)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", accent=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2", linewidth=1.4,
                                    edgecolor=SLOTS[0] if accent else BASE, facecolor="#eef4fc" if accent else "#f3f2ee"))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontsize=10, color=INK, fontweight="semibold")
        if sub:
            ax.text(x + w / 2, y + h - 4.8, sub, ha="center", va="top", fontsize=8.4, color=INK2, linespacing=1.45)

    def arrow(x1, y1, x2, y2, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12, linewidth=1.5,
                                     color=MUTED if dashed else SLOTS[0], linestyle=(0, (4, 3)) if dashed else "-"))

    try:
        n_films = len(json.loads((RESULTS.parent / "data" / "corpus" / "films.json").read_text(encoding="utf-8")))
    except (OSError, ValueError):
        n_films = "all"
    ax.text(0.5, 45, "OFFLINE  —  built once from Wikipedia", fontsize=9, color=MUTED, fontweight="bold")
    box(0.5, 31, 15, 11, "Wikipedia crawl", f"{n_films} film articles\n(wikitext, posters)")
    box(19.5, 31, 16, 11, "Pre-processing", "tokenise · fold · stop\nwords · stem")
    box(40.5, 31, 59, 11, "Offline artefacts",
        "inverted index (BM25F, TF-IDF)  ·  passage embeddings (bge-small)  ·  Naive Bayes genre/decade models\n"
        "link graph → PageRank & HITS  ·  entity lexicon (people, characters, titles, phonetic keys)")
    arrow(15.6, 36.5, 19.4, 36.5)
    arrow(35.6, 36.5, 40.4, 36.5)

    ax.text(0.5, 27.5, "ONLINE  —  every query", fontsize=9, color=MUTED, fontweight="bold")
    box(0.5, 11, 15, 13, "Your description", "“maybe the 90s,\nnot a comedy…”", accent=True)
    box(19.5, 11, 17, 13, "Clause splitter", "plain · hedged · sure\ncontext · small talk\nexclusion (scope)", accent=True)
    box(40.5, 11, 17, 13, "Hinglish layer", "names · phonetic fixes\nshaadi → wedding", accent=True)
    box(61.5, 3, 18, 21, "22 ranking features", "clause-weighted BM25F\nexclusion penalty\nTF-IDF cosine · dense\n"
        "title / cast matches\nyear · language · genre\nNB genre & decade\nPageRank · HITS · views", accent=True)
    box(83, 11, 16.5, 13, "Learning to rank", "coordinate ascent\n→ top 5 + reasons", accent=True)
    arrow(15.6, 17.5, 19.4, 17.5)
    arrow(36.6, 17.5, 40.4, 17.5)
    arrow(57.6, 17.5, 61.4, 17.5)
    arrow(79.6, 17.5, 82.9, 17.5)
    arrow(49, 30.9, 49, 24.3, dashed=True)          # entity lexicon -> Hinglish layer
    arrow(70.5, 30.9, 70.5, 24.3, dashed=True)      # index, embeddings, NB, graph -> features
    ax.text(50, 27.7, "names", fontsize=8, color=MUTED)
    ax.text(71.5, 27.7, "index · vectors · models · graph", fontsize=8, color=MUTED)
    ax.set_title("Desi ToT pipeline", fontsize=13)
    finish(fig, "fig0_pipeline.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    fig_pipeline()
    path = RESULTS / "summary.json"
    if not path.exists():
        print("results/summary.json not found: run scripts/evaluate.py first")
        return
    S = json.loads(path.read_text(encoding="utf-8"))
    for fn in (fig_main, fig_conditions, fig_success, fig_ablation, fig_sweeps, fig_phenomena, fig_negation,
               fig_confusion, fig_weights):
        try:
            fn(S)
        except (KeyError, StopIteration, ValueError) as exc:
            print(f"skipped {fn.__name__}: {exc!r}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
