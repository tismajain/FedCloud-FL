"""Controlled FedAvg/FedProx study using upstream local training, in-process.

The sequential simulator isolates algorithmic effects; it is NOT a cloud/network
benchmark. Communication is serialized tensor-payload accounting, not packet capture.
Original socket-based scripts and algorithms remain unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fedcloud.calibration import score_probabilities, select_threshold
from fedcloud.data import heterogeneity_report, make_partitions
from fedcloud.model import init_params, predict_proba, train_local
from fedcloud.serialization import payload_bytes
from fedcloud.study_data import prepare_study


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def weighted_average(updates, sizes):
    """Sample-weighted FedAvg (same weighted-sum order as Flower aggregate)."""
    if not updates or len(updates) != len(sizes) or min(sizes) <= 0:
        raise ValueError("Aggregation needs nonempty positive-sized client updates")
    n = sum(sizes)
    return [sum(w[j] * count for w, count in zip(updates, sizes)) / n
            for j in range(len(updates[0]))]


def train_run(train, validation, parts, *, rounds, seed, epochs=1, lr=.001,
              pos_weight=10., prox_mu=0.):
    """No test data in the training API. Fixed final round; no early stopping."""
    w = init_params(seed)
    sizes = [len(idx) for idx in parts]
    history, diagnostics = [], []
    bytes_total = 0
    t0 = time.perf_counter()
    train_seconds = 0.
    for rnd in range(1, rounds + 1):
        start = time.perf_counter()
        updates = []
        for cid, idx in enumerate(parts):
            local = train_local(w, train.X[idx], train.y[idx], epochs=epochs,
                                lr=lr, pos_weight=pos_weight, prox_mu=prox_mu,
                                seed=seed * 1000 + rnd * 10 + cid)
            norm = float(np.sqrt(sum(np.sum((a - b).astype(float) ** 2) for a, b in zip(local, w))))
            diagnostics.append({"round": rnd, "client": cid, "rows": len(idx),
                                "frauds": int(train.y[idx].sum()), "aggregation_weight": len(idx) / sum(sizes),
                                "update_l2": norm, "weighted_update_l2": norm * len(idx) / sum(sizes)})
            bytes_total += payload_bytes(w) + payload_bytes(local)
            updates.append(local)
        w = weighted_average(updates, sizes)
        train_seconds += time.perf_counter() - start
        prob = predict_proba(w, validation.X)
        m = score_probabilities(validation.y, prob, .5)
        history.append({"round": rnd, "communication_mb": bytes_total / 1e6,
                        "training_seconds": train_seconds, "validation": m})
        if rnd == 1 or rnd % 5 == 0 or rnd == rounds:
            print(f"  round={rnd:2d} val_AP={m['auprc']:.5f} val_F1@.5={m['f1']:.5f} MB={bytes_total/1e6:.4f}", flush=True)
    return w, history, diagnostics, {"training_seconds": train_seconds,
                                    "training_plus_validation_seconds": time.perf_counter() - t0,
                                    "communication_mb": bytes_total / 1e6,
                                    "payload_bytes_per_model": payload_bytes(w)}


def study(args):
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print("Output:", out, flush=True)
    sets, split_ids, metadata = prepare_study(args.csv, args.split_seed)
    write_json(out / "data_manifest.json", metadata)
    np.savez_compressed(out / "split_row_ids.npz", **split_ids)
    # Provenance snapshot only; these are not newly measured results.
    (out / "upstream_reported_summary.csv").write_bytes((ROOT / "results/summary.csv").read_bytes())
    print("Data:", metadata["splits"], "duplicates removed:", metadata["duplicates_removed"], flush=True)
    results, all_history, failures = [], {}, []
    train, validation, test = sets["train"], sets["validation"], sets["test"]
    configs = [("fedavg", scheme, 0.) for scheme in args.schemes]
    if args.include_fedprox:
        configs.append(("fedprox", "dir0.1", .1))
    for strategy, scheme, mu in configs:
        parts = make_partitions(train.y, args.clients, scheme, args.partition_seed)
        report = heterogeneity_report(train.y, parts)
        write_json(out / f"partitions_{scheme}.json", report)
        np.savez_compressed(out / f"partition_indices_{scheme}.npz",
                            **{f"client_{i}": idx for i, idx in enumerate(parts)})
        for seed in args.seeds:
            name = f"{strategy}_{scheme}_s{seed}"
            print(name, flush=True)
            start = time.perf_counter()
            try:
                w, history, diagnostics, cost = train_run(
                    train, validation, parts, rounds=args.rounds, seed=seed,
                    epochs=args.epochs, lr=args.lr, pos_weight=args.pos_weight, prox_mu=mu)
                # Exactly the same fitted checkpoint/scores for baseline and calibrated.
                validation_prob = predict_proba(w, validation.X)
                cal_start = time.perf_counter()
                calibration = select_threshold(validation.y, validation_prob)
                calibration_seconds = time.perf_counter() - cal_start
                # Test accessed only now, after training AND decision-rule selection.
                test_prob = predict_proba(w, test.X)
                np.savez_compressed(out / f"{name}_predictions.npz", validation_y=validation.y,
                                    validation_prob=validation_prob, test_y=test.y, test_prob=test_prob)
                np.savez_compressed(out / f"{name}_model.npz", **{f"param_{j}": p for j, p in enumerate(w)})
                pd.DataFrame(diagnostics).to_csv(out / f"{name}_client_diagnostics.csv", index=False)
                diagnosis = {"validation_score_quantiles": {
                    str(label): np.quantile(validation_prob[validation.y == label], [0, .25, .5, .75, 1]).tolist()
                    for label in (0, 1)}, "calibration": calibration,
                    "calibration_seconds": calibration_seconds,
                    "total_run_seconds": time.perf_counter() - start}
                write_json(out / f"{name}_diagnostics.json", diagnosis)
                all_history[name] = history
                for method, threshold in (("fixed_0.5", .5), ("validation_f1", calibration["threshold"])):
                    measured = score_probabilities(test.y, test_prob, threshold)
                    row = {"run": name, "strategy": strategy, "scheme": scheme,
                           "seed": seed, "method": method, "rounds": args.rounds,
                           "epochs": args.epochs, "pos_weight": args.pos_weight, "prox_mu": mu,
                           **cost, "calibration_seconds": calibration_seconds if method == "validation_f1" else 0.,
                           "additional_training_mb": 0., **measured}
                    results.append(row)
                    print(f"  TEST {method}: AP={row['auprc']:.5f}, F1={row['f1']:.5f}, recall={row['recall']:.5f}, threshold={threshold:.8g}", flush=True)
            except Exception as exc:
                failures.append({"run": name, "error": f"{type(exc).__name__}: {exc}"})
                write_json(out / "failures.json", failures)
                raise
            finally:
                pd.DataFrame(results).to_csv(out / "results.csv", index=False)
                write_json(out / "history.json", all_history)
    df = pd.DataFrame(results)
    measured = ["auprc", "f1", "precision", "recall", "roc_auc", "communication_mb", "training_seconds", "threshold"]
    summary = df.groupby(["strategy", "scheme", "method"])[measured].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.to_csv(out / "summary.csv")
    write_json(out / "failures.json", failures)
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "flwr", "pytest", "threadpoolctl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    files = ["fedcloud/model.py", "fedcloud/data.py", "fedcloud/calibration.py",
             "fedcloud/study_data.py", "scripts/run_threshold_study.py"]
    config = {**vars(args), "csv": str(args.csv), "output": str(out),
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "file_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files},
              "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
              "execution": "sequential in-process federated simulation; full participation; BLAS threads=1",
              "communication": "Decimal MB; exact serialized tensor sizes, simulated transfers; excludes transport/control overhead",
              "threshold_distribution": "Server-side threshold only; broadcasting one float64 to five clients would add 40 payload bytes",
              "test_policy": "Fixed final checkpoint; test used only after validation threshold is frozen"}
    write_json(out / "config.json", config)
    from plot_threshold_study import make_figures
    make_figures(out)
    print("Completed", out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/creditcard.csv")
    parser.add_argument("--output", default=str(ROOT / "results" / datetime.now(timezone.utc).strftime("minor_%Y%m%dT%H%M%SZ")))
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--pos-weight", type=float, default=10.)
    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--schemes", nargs="+", default=["dir0.1", "dir0.5"])
    parser.add_argument("--include-fedprox", action="store_true")
    args = parser.parse_args()
    if min(args.rounds, args.epochs, args.clients) < 1 or args.lr <= 0 or args.pos_weight <= 0:
        parser.error("Training parameters must be positive")
    with threadpool_limits(limits=1):
        study(args)


if __name__ == "__main__":
    main()
