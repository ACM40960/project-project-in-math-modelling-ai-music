# tools/model_compare.py
# Usage:
#   python tools/model_compare.py --metrics artifacts/metrics_latest.json [--dark]
# Outputs:
#   figures/model_compare.png (transparent) and figures/model_compare.svg

import json, argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True,
                    help="Path to metrics JSON written by eval_report.py")
    ap.add_argument("--dark", action="store_true",
                    help="White text for dark posters")
    args = ap.parse_args()

    # Read JSON
    data = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    models = list(data["models"].keys())
    metric_names = ["Accuracy", "F1-macro", "F1-weighted"]

    # Colors
    COL_ACC = "#14B8A6"
    COL_F1M = "#6366F1"
    COL_F1W = "#38BDF8"
    colors = [COL_ACC, COL_F1M, COL_F1W]

    # Data array
    arr = np.array([[data["models"][m][k] for k in metric_names] for m in models])

    # Force color theme
    if args.dark:
        text_col = "#FFFFFF"
        edge_col = "#E5E7EB"
        grid_col = "#9CA3AF"
    else:
        text_col = "#000000"
        edge_col = "#111827"
        grid_col = "#9CA3AF"

    plt.rcParams.update({
        "figure.facecolor": "none",
        "axes.facecolor":   "none",
        "axes.edgecolor":   edge_col,
        "xtick.color":      text_col,
        "ytick.color":      text_col,
        "text.color":       text_col,
        "grid.color":       grid_col,
        "axes.labelcolor":  text_col,
        "legend.labelcolor": text_col
    })

    # Layout
    n_models = len(models)
    bar_width = 0.22
    x = np.arange(n_models)
    fig_w = max(6.5, 3.0 + 1.6 * n_models)
    fig_h = 4.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # Bars
    for i, name in enumerate(metric_names):
        ax.bar(
            x + (i - 1) * bar_width,
            arr[:, i],
            width=bar_width,
            label=name,
            color=colors[i],
            edgecolor="none",
        )

    ymax = max(0.65, float(arr.max()) * 1.12)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.linspace(0, 1.0, 6))
    ax.set_yticklabels([f"{int(t*100)}%" for t in np.linspace(0, 1.0, 6)], color=text_col)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="center", color=text_col)
    ax.set_ylabel("Score", color=text_col)

    # Title
    title = "Text Emotion Classification — Validation Metrics"
    if "val_size" in data:
        title += f"  (n={data['val_size']})"
    ax.set_title(title, pad=10, color=text_col)

    # Remove top/right spines
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    # Value labels
    for i in range(arr.shape[1]):
        for j in range(arr.shape[0]):
            v = arr[j, i]
            ax.text(x[j] + (i - 1) * bar_width, v + 0.01,
                    f"{v*100:.1f}%", ha="center", va="bottom",
                    fontsize=9, color=text_col)

    ax.legend(frameon=False, ncols=3, loc="upper left", bbox_to_anchor=(0, 1.02), labelcolor=text_col)
    fig.tight_layout()

    # Save
    Path("figures").mkdir(exist_ok=True)
    fig.savefig("figures/model_compare.png", transparent=True, bbox_inches="tight", dpi=300)
    fig.savefig("figures/model_compare.svg", bbox_inches="tight")
    print("Saved figures/model_compare.png and figures/model_compare.svg")

if __name__ == "__main__":
    main()
