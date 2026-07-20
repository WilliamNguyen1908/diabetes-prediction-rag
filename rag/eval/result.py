"""Visualize the evaluation results (retrieval + generator) in one figure.

Reads the saved eval outputs and renders three panels:
  1. Retrieval quality — hybrid vs dense (coverage, recall@k, hit@k, MRR).
  2. Safety guarantees — stat tiles (ungrounded drugs reaching user, fabricated
     doses, drug grounding, overall faithfulness).
  3. Generation quality by pillar — faithfulness vs answer relevancy.

Run:  uv run python rag/eval/result.py
Output: rag/eval/results.png
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"   # eval outputs now live in eval/results/

# --- validated categorical + status palette (from the dataviz reference) ---
BLUE, AQUA = "#2a78d6", "#1baf7a"        # categorical slots 1 & 2
GOOD, CRIT = "#0ca30c", "#d03b3b"        # status
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, SURFACE = "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": MUTED,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
})


def load(name):
    return json.loads((RESULTS_DIR / name).read_text(encoding="utf-8"))


def style_axis(ax, ymax=1.0):
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(length=0)


def grouped_bars(ax, groups, s1, s2, l1, l2, c1, c2):
    """Two-series grouped bars with direct value labels."""
    x = np.arange(len(groups))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, s1, w, label=l1, color=c1, zorder=3)
    b2 = ax.bar(x + w / 2 + 0.01, s2, w, label=l2, color=c2, zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)


def stat_tile(ax, value, label, color, sub=""):
    ax.axis("off")
    ax.text(0.5, 0.62, value, ha="center", va="center", fontsize=30, fontweight="bold", color=color)
    ax.text(0.5, 0.24, label, ha="center", va="center", fontsize=10.5, color=INK)
    if sub:
        ax.text(0.5, 0.06, sub, ha="center", va="center", fontsize=8.5, color=MUTED)


def main():
    ret = load("retrieval_results.json")
    gen = load("generator_results.json")["summary"]
    judge = load("judge_results.json")

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.15, 0.55, 1.25], hspace=0.7, wspace=0.4,
                          left=0.07, right=0.97, top=0.84, bottom=0.08)
    fig.suptitle("Diabetes RAG — Evaluation Results", x=0.07, y=0.965, ha="left",
                 fontsize=17, fontweight="bold", color=INK)
    fig.text(0.07, 0.925, "Retrieval quality · safety guarantees · generation quality by clinical pillar",
             ha="left", fontsize=10.5, color=MUTED)

    # --- Panel 1: retrieval hybrid vs dense ---
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ["coverage", "recall", "hit", "mrr"]
    labels = ["coverage", "recall@k", "hit@k", "MRR"]
    hy = [ret["hybrid"][m] for m in metrics]
    de = [ret["dense"][m] for m in metrics]
    grouped_bars(ax1, labels, hy, de, "Hybrid (dense+BM25+RRF)", "Dense only", BLUE, AQUA)
    style_axis(ax1)
    ax1.set_title(f"Retrieval quality (k={ret['k']}, 43 questions)", fontsize=12,
                  fontweight="bold", color=INK, loc="left", pad=26)
    ax1.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)

    # --- Panel 2: overall generation metrics ---
    ax2 = fig.add_subplot(gs[0, 2:])
    over = ["faithfulness", "answer_relevancy"]
    ov_vals = [judge["summary"]["faithfulness"], judge["summary"]["answer_relevancy"]]
    b = ax2.bar([0, 1], ov_vals, 0.5, color=[BLUE, AQUA], zorder=3)
    for r in b:
        ax2.annotate(f"{r.get_height():.2f}", (r.get_x() + r.get_width() / 2, r.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK2)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["faithfulness", "answer\nrelevancy"])
    style_axis(ax2)
    ax2.set_title(f"Generation, overall (judge: {judge['summary']['judge']})", fontsize=12,
                  fontweight="bold", color=INK, loc="left", pad=10)

    # --- Panel row 2: safety stat tiles ---
    reach = int(gen.get("ungrounded_reaching_user", 0))
    dose = int(gen.get("answers_with_a_dose", 0))
    tiles = [
        (str(reach), "ungrounded drugs\nreaching the user", GOOD if reach == 0 else CRIT, "safety filter"),
        (str(dose), "answers giving\na drug dose", GOOD if dose == 0 else CRIT, "prompt forbids dosing"),
        (f"{gen['drug_grounding_rate']:.2f}", "drug grounding\n(raw model)", INK, "agent or class in context"),
        (f"{judge['summary']['faithfulness']:.2f}", "overall\nfaithfulness", INK, "claims backed by context"),
    ]
    for i, (val, lab, col, sub) in enumerate(tiles):
        stat_tile(fig.add_subplot(gs[1, i]), val, lab, col, sub)

    # --- Panel 3: faithfulness & relevancy by pillar ---
    ax3 = fig.add_subplot(gs[2, :])
    byp = defaultdict(lambda: {"f": [], "r": []})
    for row in judge["rows"]:
        if row["faithfulness"] is not None:
            byp[row["pillar"]]["f"].append(row["faithfulness"])
        if row["answer_relevancy"] is not None:
            byp[row["pillar"]]["r"].append(row["answer_relevancy"])
    # order by relevancy desc to tell the "designed-for pillars score high" story
    order = sorted(byp, key=lambda p: -np.mean(byp[p]["r"]))
    faith = [np.mean(byp[p]["f"]) for p in order]
    relev = [np.mean(byp[p]["r"]) for p in order]
    xlabels = [f"{p}\n(n={len(byp[p]['f'])})" for p in order]
    grouped_bars(ax3, xlabels, faith, relev, "faithfulness", "answer relevancy", BLUE, AQUA)
    style_axis(ax3)
    ax3.set_title("Generation quality by clinical pillar  —  relevancy is high on the "
                  "recommendation pillars (diet/lifestyle), low on narrow Q&A pillars",
                  fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=26)
    ax3.legend(frameon=False, fontsize=9, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)

    out = EVAL_DIR / "results.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
