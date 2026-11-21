"""Plot only measured study artifacts; no synthetic or manually typed metrics."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def make_figures(out):
    out = Path(out)
    os.environ.setdefault("MPLCONFIGDIR", str(out / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(out / "results.csv")
    groups = list(df.groupby(["strategy", "scheme", "method"], sort=False))
    labels = ["\n".join(k) for k, _ in groups]
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), layout="constrained")
    for ax, metric in zip(axes, ("auprc", "f1", "recall")):
        means = [sub[metric].mean() for _, sub in groups]
        stds = [sub[metric].std(ddof=1) if len(sub) > 1 else 0 for _, sub in groups]
        ax.bar(range(len(groups)), means, yerr=stds, capsize=3)
        ax.set_xticks(range(len(groups)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(metric.upper() + " (mean ± sample SD across model seeds)")
    fig.savefig(out / "comparison.png", dpi=160)
    plt.close(fig)

    seed = int(df.seed.min())
    selected = df[df.seed == seed]
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for run, rows in selected.groupby("run"):
        data = np.load(out / f"{run}_predictions.npz")
        precision, recall, _ = precision_recall_curve(data["test_y"], data["test_prob"])
        ax.plot(recall, precision, label=f"{run} AP={rows.auprc.iloc[0]:.3f}")
        for _, row in rows.iterrows():
            ax.scatter(row.recall, row.precision, marker="o" if row.method == "fixed_0.5" else "x")
    ax.axhline(float(data["test_y"].mean()), color="gray", linestyle="--", label="Test prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Test precision-recall curves, seed {seed}\nSame curve before/after threshold selection; o=fixed, x=validation")
    ax.legend(fontsize=8)
    fig.savefig(out / "precision_recall.png", dpi=160)
    plt.close(fig)

    n = len(selected)
    fig, axes = plt.subplots(2, int(np.ceil(n / 2)), figsize=(4 * int(np.ceil(n / 2)), 8),
                             squeeze=False, layout="constrained")
    for ax, (_, row) in zip(axes.flat, selected.iterrows()):
        cm = np.array([[row.tn, row.fp], [row.fn, row.tp]], dtype=int)
        ax.imshow(np.log1p(cm), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if np.log1p(cm[i,j]) > np.log1p(cm.max())/2 else "black")
        ax.set_xticks([0, 1], ["Legit", "Fraud"])
        ax.set_yticks([0, 1], ["Legit", "Fraud"])
        ax.set(xlabel="Predicted", ylabel="Actual",
               title=f"{row.strategy} {row.scheme}\n{row.method}, t={row.threshold:.4g}")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"Test confusion matrices, seed {seed}; counts shown (log colour scale)")
    fig.savefig(out / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    history = json.loads((out / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for run, rows in history.items():
        if not run.endswith(f"_s{seed}"):
            continue
        axes[0].plot([r["communication_mb"] for r in rows],
                     [r["validation"]["auprc"] for r in rows], label=run)
    axes[0].set(xlabel="Cumulative model payload MB (simulated)", ylabel="Validation AUPRC",
                title="Training progress; final test not used for checkpoint selection")
    axes[0].legend(fontsize=8)
    for (strategy, scheme, method), sub in groups:
        axes[1].scatter(sub.communication_mb.mean(), sub.f1.mean(),
                        marker="o" if method == "fixed_0.5" else "x",
                        label=f"{strategy} {scheme} {method}", s=70)
    axes[1].set(xlabel="Training model payload MB (simulated)", ylabel="Mean final-test F1",
                title="Threshold selection adds no training transfers")
    axes[1].legend(fontsize=7)
    fig.savefig(out / "performance_communication.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    make_figures(parser.parse_args().directory)
