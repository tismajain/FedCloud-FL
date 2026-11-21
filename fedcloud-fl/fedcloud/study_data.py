"""Leakage-resistant data preparation for the NEW controlled threshold study.

Original data.load_and_preprocess and on-disk partitions are unchanged. The study
deduplicates identical feature vectors before a 60/20/20 stratified row split and
fits standardisation only on the 60% training rows. No customer IDs are available.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import FEATURES, LABEL, Split


def prepare_study(csv_path, seed=42):
    df = pd.read_csv(csv_path)
    if not set(FEATURES + [LABEL]).issubset(df.columns):
        raise ValueError("CSV does not have the expected fraud dataset schema")
    if not np.isin(df[LABEL], [0, 1]).all():
        raise ValueError("Non-binary fraud labels")
    n_original, original_frauds = len(df), int(df[LABEL].sum())
    # Reject feature-identical records with contradictory labels rather than hide them.
    duplicates = df[df.duplicated(FEATURES, keep=False)]
    if len(duplicates) and duplicates.groupby(FEATURES, dropna=False)[LABEL].nunique().max() > 1:
        raise ValueError("Feature-identical transactions have conflicting labels")
    df = df.drop_duplicates(FEATURES, keep="first")
    if not np.isfinite(df[FEATURES].to_numpy()).all() or (df.Amount < 0).any():
        raise ValueError("Invalid numeric input")
    ids = df.index.to_numpy(dtype=np.int64)
    labels = df[LABEL].to_numpy(dtype=np.float32)
    positions = np.arange(len(df))
    development, test = train_test_split(positions, test_size=.2, stratify=labels, random_state=seed)
    train, validation = train_test_split(development, test_size=.25,
                                         stratify=labels[development], random_state=seed)
    raw = df[FEATURES].copy()
    raw["Amount"] = np.log1p(raw["Amount"])
    X = raw.to_numpy(dtype=np.float32)
    mean, std = X[train].mean(axis=0), X[train].std(axis=0) + 1e-6
    sets, split_ids = {}, {}
    for name, idx in (("train", train), ("validation", validation), ("test", test)):
        sets[name] = Split(((X[idx] - mean) / std).astype(np.float32), labels[idx])
        split_ids[name] = ids[idx]
    assert not (set(split_ids["train"]) & set(split_ids["validation"]))
    assert not (set(split_ids["train"]) & set(split_ids["test"]))
    assert not (set(split_ids["validation"]) & set(split_ids["test"]))
    metadata = {"source": str(Path(csv_path).name),
                "sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                "split_seed": seed, "original_rows": n_original,
                "original_frauds": original_frauds, "deduplicated_rows": len(df),
                "duplicates_removed": n_original - len(df),
                "features": FEATURES, "scaler_mean": mean.tolist(), "scaler_std": std.tolist(),
                "splits": {k: {"rows": len(s), "frauds": int(s.y.sum())} for k, s in sets.items()},
                "split_unit": "unique transaction feature vector, not customer or chronological split"}
    return sets, split_ids, metadata
