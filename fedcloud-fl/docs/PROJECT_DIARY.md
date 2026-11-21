# Project diary

Entries reflect Git timestamps and saved evidence. They record checkpoints with known outcomes.

| Recorded date | Evidence | Verified activity and outcome |
|---|---|---|
| 2026-11-21 11:17 UTC | [Study configuration](../results/minor_calibration_20261121/config.json), [results](../results/minor_calibration_20261121/results.csv) | The configuration records five clients, model seeds 0/1/2 and split/partition seed 42. The saved CSV contains nine runs and 18 paired observations. |
| 2026-11-21 11:20 UTC | `2d3794f` | Committed the threshold selector, controlled-study harness, calibration tests and measured checkpoint. |
| 2026-11-21 17:26 +05:30 | `3ca7895`, [artifact audit](EXPERIMENTS.md#artifact-audit) | Audited saved figures and result arithmetic. Added an option to run the upstream socket wrapper with WAN emulation. No independent socket measurements were recorded. |
| 2026-11-21 17:35 +05:30 | `f7016f8`, [ranking checkpoint](PROGRESS.md#ranking-verification-checkpoint) | Added the prediction-based ranking verifier. Synthetic cases covered ties, a changed metric and a missing archive. Real prediction verification was pending. |
| 2026-11-21 17:43 +05:30 | `3122b9a`, [environment checkpoint](PROGRESS.md#clean-environment-retry) | Created an isolated Python 3.12.14 environment. PyPI connection failure blocked dependency installation. Pytest failed to start and unittest discovery reported two missing-NumPy errors. The Flower parity test now exposes transitive import errors. |
| 2026-11-21 | `f8d8ece`, [project report](MINOR_PROJECT_REPORT.md), [demonstration](DEMONSTRATION.md) | Expanded the report and documented the exact nine-run procedure. Cross-checked all six result rows against the archived summary. No new measurements were generated. |
