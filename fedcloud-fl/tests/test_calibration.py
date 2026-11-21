"""New scientific-contract tests; upstream smoke tests are retained unchanged."""
import inspect
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fedcloud.calibration import score_probabilities, select_threshold
from fedcloud.data import FEATURES, Split
from fedcloud.model import predict_proba
from fedcloud.study_data import prepare_study
from run_threshold_study import train_run, weighted_average


class CalibrationTests(unittest.TestCase):
    def test_known_optimal_threshold(self):
        y = [0, 1, 0, 1]
        p = [.01, .1, .02, .2]
        chosen = select_threshold(y, p)
        self.assertEqual(chosen["threshold"], .1)
        self.assertEqual(chosen["validation_f1"], 1.)
        self.assertEqual(score_probabilities(y, p, .5)["f1"], 0.)

    def test_threshold_does_not_change_ranking_metrics(self):
        y, p = [0, 0, 1, 1], [.1, .2, .3, .4]
        fixed = score_probabilities(y, p, .5)
        tuned = score_probabilities(y, p, select_threshold(y, p)["threshold"])
        self.assertEqual(fixed["auprc"], tuned["auprc"])
        self.assertEqual(fixed["roc_auc"], tuned["roc_auc"])
        self.assertEqual((tuned["tn"], tuned["fp"], tuned["fn"], tuned["tp"]), (2, 0, 0, 2))

    def test_tied_scores_match_exhaustive_search(self):
        rng = np.random.default_rng(12)
        for _ in range(20):
            y = np.r_[0, 1, rng.integers(0, 2, 30)]
            p = np.round(rng.random(len(y)), 1)
            candidates = [(score_probabilities(y, p, t)["f1"], t) for t in np.unique(p)]
            best_f1, best_threshold = max(candidates)
            selected = select_threshold(y, p)
            self.assertAlmostEqual(selected["validation_f1"], best_f1)
            self.assertAlmostEqual(selected["threshold"], best_threshold)

    def test_single_class_and_invalid_input(self):
        for y, p in (([], []), ([0, 1], [.2]), ([0, 2], [.1, .8]),
                     ([0, 1], [np.nan, .2]), ([0, 1], [-.1, .2])):
            with self.assertRaises(ValueError):
                select_threshold(y, p)
        with self.assertRaises(ValueError):
            select_threshold([0, 0], [.1, .2])
        scored = score_probabilities([0, 0], [.1, .2])
        self.assertIsNone(scored["roc_auc"])
        self.assertIsNone(scored["auprc"])

    def test_test_labels_cannot_affect_validation_selection(self):
        y, p = np.array([0, 0, 1, 1]), np.array([.02, .04, .06, .08])
        chosen = select_threshold(y, p)
        for test_y in ([0, 0, 0, 1], [1, 1, 1, 0]):
            score_probabilities(test_y, [.2, .3, .4, .5], chosen["threshold"])
            self.assertEqual(chosen, select_threshold(y, p))
        self.assertNotIn("test", inspect.signature(train_run).parameters)

    def test_preprocessing_split_and_duplicates(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(300, 30)), columns=FEATURES)
        frame["Amount"] = np.abs(frame.Amount)
        frame["Class"] = np.arange(300) % 10 == 0
        frame = pd.concat([frame, frame.iloc[:10]], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.csv"
            frame.to_csv(path, index=False)
            sets, ids, meta = prepare_study(path)
            self.assertEqual(meta["duplicates_removed"], 10)
            self.assertEqual(sum(map(len, ids.values())), 300)
            self.assertEqual(len(np.unique(np.concatenate(list(ids.values())))), 300)
            raw = frame.loc[ids["train"], FEATURES].copy()
            raw["Amount"] = np.log1p(raw.Amount)
            np.testing.assert_allclose(meta["scaler_mean"], raw.to_numpy(dtype=np.float32).mean(0), atol=1e-7)
            np.testing.assert_allclose(sets["train"].X.mean(0), 0, atol=2e-6)
            again, ids2, _ = prepare_study(path)
            for key in ids:
                np.testing.assert_array_equal(ids[key], ids2[key])
                np.testing.assert_array_equal(sets[key].X, again[key].X)

    def test_aggregation_and_training_reproducible(self):
        a, b = [np.ones((2, 2), dtype=np.float32)], [np.zeros((2, 2), dtype=np.float32)]
        np.testing.assert_array_equal(weighted_average([a, b], [1, 3])[0], .25 * a[0])
        rng = np.random.default_rng(3)
        X = rng.normal(size=(100, 30)).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.float32)
        parts = [np.arange(50), np.arange(50, 100)]
        split = Split(X, y)
        w1, h1, _, cost = train_run(split, split, parts, rounds=2, seed=1)
        w2, h2, _, _ = train_run(split, split, parts, rounds=2, seed=1)
        for p, q in zip(w1, w2):
            np.testing.assert_array_equal(p, q)
        self.assertEqual(h1[-1]["validation"], h2[-1]["validation"])
        self.assertEqual(cost["communication_mb"], 2 * 2 * 2 * cost["payload_bytes_per_model"] / 1e6)
        self.assertTrue(np.isfinite(predict_proba(w1, X)).all())

    def test_aggregation_matches_flower(self):
        try:
            from flwr.server.strategy.aggregate import aggregate
        except ModuleNotFoundError as exc:
            if exc.name != "flwr":
                raise
            self.skipTest("Flower is absent; install the project requirements to verify parity")
        rng = np.random.default_rng(4)
        updates = [[rng.normal(size=(3, 4)).astype(np.float32), np.ones(4, dtype=np.float32)] for _ in range(5)]
        sizes = [3, 10, 5, 8, 2]
        for actual, expected in zip(weighted_average(updates, sizes), aggregate(list(zip(updates, sizes)))):
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
