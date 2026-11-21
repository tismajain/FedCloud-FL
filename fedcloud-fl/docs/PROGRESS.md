# FedCloud-FL progress and handoff

Checkpoint date: 2026-11-21. This is a working experimental prototype, **not a completed academic submission**.

## Location and Git policy

- Local Git root: `/mnt/c/Users/Tisma Jain/cc`; application: `cc/fedcloud-fl`.
- Working branch: `feature/non-iid-fraud-improvements`.
- Intended private destination: `https://github.com/tismajain/FedCloud-FL`.
- Original source: `https://github.com/deepankitasaha/cc`, revision `3ba5dd0c0950e665f2bf6ac88d57e4c0c547fe2e`.
- Upstream history contains transaction dataset archives. The publishable branch is a clean source snapshot, with original history retained on the local `local/prepublish-history` branch and the original remote retained as `upstream`. Do not push that historical branch.
- Raw/partition data, NPZ predictions/models/split IDs, virtual environments, and credentials are excluded. Reproduction regenerates these locally. Small measured CSV/JSON results and PNG figures are saved to Git.
- Commit and push each completed milestone; never force-push.

## Implemented

- `fedcloud/calibration.py`: validation-only F1 threshold search, deterministic handling of tied scores, validation of inputs, confusion counts and imbalance-aware metrics. Does not change score ranking or claim probability calibration.
- `fedcloud/study_data.py`: remove feature-identical duplicates, reject conflicting duplicate labels, fixed stratified 60/20/20 split, training-only scaling after log Amount transformation, auditable split IDs and dataset hash.
- `scripts/run_threshold_study.py`: deterministic sequential FedAvg/FedProx simulation reusing upstream `train_local`, shared final checkpoint for fixed and calibrated thresholds, validation-only training monitoring, isolated result directories, payload accounting and client diagnostics.
- `scripts/plot_threshold_study.py`: comparison, precision-recall, confusion matrix and communication-performance plots from measured files.
- `scripts/reproduce_socket_baseline.py`: wrapper for unchanged upstream socket processes, isolated logs and bounded runtime. Added but not yet exercised at this checkpoint.
- `tests/test_calibration.py`: eight new tests covering thresholds, ties, invalid inputs, leakage boundaries, preprocessing, deterministic training, communication arithmetic, and Flower aggregation parity.
- Root `.gitignore`: exclude datasets, large binary artifacts, secrets, environments.
- Original FedAvg, FedProx, FedAsync, centralized code and upstream results are unchanged.

## Tests and setup completed

- Original `python tests/test_smoke.py`: all five tests passed before changes.
- New unittest suite: seven passed, one skipped before Flower installation.
- Combined `python -m pytest tests -q`: **12 passed, 1 skipped**, 4 dependency deprecation warnings. Flower parity currently skips on ImportError; investigate before claiming transport verification.
- Dataset downloaded and original alpha=0.1 partition created using upstream `scripts/prepare_data.py`; 284,807 rows, 492 frauds, 227,845 original training rows, 56,962 original test rows (98 frauds).
- Controlled study: 1,081 identical duplicates removed, leaving 283,726 rows: train 170,235 (284 frauds), validation 56,745 (94), test 56,746 (95).
- Nine 30-round training runs completed, five full-participation clients, one local epoch, LR .001, positive BCE weight 10, model seeds 0/1/2, split and partition seeds 42. FedProx mu=.1. No failed study runs.

## Actual results

Source: `results/minor_calibration_20261121/results.csv` and `summary.csv`. Means across three model seeds on one fixed split and partition per alpha; not three independent datasets. Each run accounts for **5.1468 MB** serialized model payload. Sequential simulated communication is not packet capture or cloud performance.

| Method | Alpha | Decision threshold | AUPRC | F1 | Precision | Recall | ROC-AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| FedAvg | .1 | fixed .5 | .648978 | .451899 | .983333 | .298246 | .890921 |
| FedAvg | .1 | validation F1 | .648978 | .693133 | .816411 | .603509 | .890921 |
| FedAvg | .5 | fixed .5 | .771264 | .757164 | .813472 | .712281 | .941525 |
| FedAvg | .5 | validation F1 | .771264 | .774526 | .874019 | .698246 | .941525 |
| FedProx | .1 | fixed .5 | .699607 | .661709 | .863487 | .536842 | .932699 |
| FedProx | .1 | validation F1 | .699607 | .744125 | .832668 | .673684 | .932699 |

FedAvg alpha .1 selected thresholds: 0.0000015890326, 0.000095891963, 0.000016174727. Mean training time ~9.91 s (sequential local CPU execution, excludes per-round validation). Calibration leaves AUPRC and ROC-AUC exactly unchanged. Alpha .5 seed 2 has a small F1 decrease (.790419 to .787879); calibration is not guaranteed to improve held-out results. Precision decreases for alpha .1 when recall increases.

Upstream **reported**, not yet independently reproduced: original `results/summary.csv` gives FedAvg alpha .1 AUPRC .07566 and F1 0 across three seeds. Do not compare that directly with .648978 as an improvement: the new split, deduplication and resulting client partition differ. Original partition has 393/394 frauds on one client, while a zero-fraud client holds 183,620/227,845 rows. New alpha .1 partition is different and less concentrated (fraud CV 1.264 vs original 1.994).

## Known blockers / failures

- Sandbox DNS blocked clone, dataset and pip calls; authorized network retries succeeded.
- First Flower dependency resolution failed on cryptography; explicit public PyPI retry succeeded.
- To avoid slow large-package downloads, `/tmp/fedcloud-venv` currently reuses scientific packages from a read-only pre-existing Python 3.12 environment via a `.pth` file. Flower and pytest were installed in the new environment. That inherited environment exposes TensorFlow, whose protobuf requirement conflicts with Flower's protobuf<5 constraint. Remove this inheritance by copying only required dependencies into an isolated environment, then rerun tests and socket experiments. Do not alter the unrelated source environment.
- No full upstream socket reproduction has completed yet. The new study is explicitly an in-process federated simulation.
- No customer IDs or chronological holdout; deduplication does not establish customer-level independence. PCA features were supplied precomputed. No formal privacy guarantee, secure aggregation or differential privacy.

## Remaining tasks

1. Verify private GitHub checkpoint push and record URL/commit in handoff.
2. Isolate dependencies, rerun all tests without the Flower skip, save a requirements lock.
3. Reproduce original FedAvg alpha .1 through the socket wrapper with isolated logs; optionally smoke-test upstream async and centralized paths.
4. Inspect all four plots and audit artifacts; add exact baseline reproduction results.
5. Finish `MINOR_PROJECT_REPORT.md`, `EXPERIMENTS.md`, `CONTRIBUTIONS.md`, `PRESENTATION_OUTLINE.md`, and `PROJECT_DIARY_TEMPLATE.md` with verified references and limitations.
6. Update this document, commit and push the next milestone. Verify clean status and remote SHA.
7. Before academic submission: independent partition seeds/time splits, uncertainty estimates, operational costs, broader literature review and supervisor/institution formatting review. Thresholding alone does not solve primary-metric ranking degradation.

## Resume commands

```bash
git clone --branch feature/non-iid-fraud-improvements https://github.com/tismajain/FedCloud-FL.git
cd FedCloud-FL/fedcloud-fl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --index-url https://pypi.org/simple -r requirements.txt pytest
OPENBLAS_NUM_THREADS=1 python -m pytest tests -q
python scripts/prepare_data.py --scheme dir0.1 --out data/minor_original_dir0.1
OPENBLAS_NUM_THREADS=1 python scripts/run_threshold_study.py --rounds 30 --seeds 0 1 2 --include-fedprox --output results/minor_resumed
python scripts/reproduce_socket_baseline.py --parts data/minor_original_dir0.1 --rounds 30 --seed 0 --output results/minor_socket_resumed
```

Use a fresh output directory each run; the new runners refuse to overwrite one. On this machine the current interpreter is `/tmp/fedcloud-venv/bin/python`. Read this progress file before rerunning completed experiments. Binary prediction archives needed to regenerate PR plots exist locally but are intentionally not pushed; rerunning the study regenerates them.
