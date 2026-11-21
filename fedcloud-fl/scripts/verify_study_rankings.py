"""Recompute test average precision and ROC AUC from a study's prediction archives."""
import argparse
import csv
import math
from pathlib import Path

import numpy as np


def ranking_metrics(labels, scores):
    if labels.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("expected matching one-dimensional labels and scores")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("expected binary labels and finite scores")
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("both classes must be present")

    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    # Advance the retrieval curve once per distinct score, including all ties.
    group_ends = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(scores)]
    true_positives = np.cumsum(labels[order], dtype=np.int64)[group_ends - 1]
    false_positives = group_ends - true_positives
    new_positives = np.diff(true_positives, prepend=0)
    new_negatives = np.diff(false_positives, prepend=0)
    average_precision = np.sum(new_positives * (true_positives / group_ends)) / positives
    # Each positive outranks lower-scoring negatives and gets half credit for ties.
    roc_auc = np.sum(new_positives * (negatives - false_positives + new_negatives / 2))
    return {"auprc": float(average_precision), "roc_auc": float(roc_auc / (positives * negatives))}


def verify(study):
    with (study / "results.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("results.csv contains no measurements")

    by_run = {}
    for row in rows:
        by_run.setdefault(row["run"], []).append(row)

    for run, measurements in by_run.items():
        if sorted(row["method"] for row in measurements) != ["fixed_0.5", "validation_f1"]:
            raise ValueError(f"{run}: expected one measurement for each threshold method")
        with np.load(study / f"{run}_predictions.npz", allow_pickle=False) as archive:
            labels = archive["test_y"]
            metrics = ranking_metrics(labels, archive["test_prob"])
        for row in measurements:
            if int(row["n"]) != len(labels) or int(row["frauds"]) != labels.sum():
                raise ValueError(f"{run}/{row['method']}: test counts differ from archive")
            for metric, actual in metrics.items():
                recorded = float(row[metric])
                if not math.isclose(recorded, actual, rel_tol=0, abs_tol=1e-12):
                    raise ValueError(
                        f"{run}/{row['method']}: {metric} recorded {recorded}, recomputed {actual}"
                    )
    return len(by_run), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path, help="Directory containing results.csv and prediction archives")
    args = parser.parse_args()
    try:
        runs, rows = verify(args.study)
    except (OSError, KeyError, ValueError) as exc:
        parser.exit(1, f"Verification failed: {exc}\n")
    print(f"Verified test average precision and ROC AUC for {runs} runs and {rows} measurements.")


if __name__ == "__main__":
    main()
