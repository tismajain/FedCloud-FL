# Non-IID fraud detection experiments

## Controlled threshold study

The input was the ULB Credit Card Fraud Detection dataset. The saved manifest records the CSV SHA-256 and 1,081 duplicate feature vectors removed before stratification. The train, validation and test splits contain 170,235, 56,745 and 56,746 transactions respectively, including 284, 94 and 95 frauds. The scaler was fitted on training data after applying log1p to Amount. [Dataset listing](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) · [Data manifest](../results/minor_calibration_20261121/data_manifest.json)

Five clients participated in every round of a sequential in-process simulation. The design used 30 rounds, one local epoch, learning rate 0.001, positive BCE weight 10, split and partition seeds 42, and model seeds 0, 1 and 2. Conditions were FedAvg at alpha 0.1 and 0.5 and FedProx at alpha 0.1 with mu 0.1. Each final model produced one set of test scores evaluated at 0.5 and at the threshold maximizing validation F1. Ties favor the higher threshold. [Run configuration](../results/minor_calibration_20261121/config.json)

| Condition | Mean test AUPRC (average precision) | Mean F1 at 0.5 | Mean F1 at validation threshold |
|---|---:|---:|---:|
| FedAvg, alpha 0.1 | 0.6490 | 0.4519 | 0.6931 |
| FedAvg, alpha 0.5 | 0.7713 | 0.7572 | 0.7745 |
| FedProx, alpha 0.1 | 0.6996 | 0.6617 | 0.7441 |

AUPRC and ROC-AUC were identical across paired thresholds in every run. At FedAvg alpha 0.1, mean recall rose from 0.2982 to 0.6035 and mean precision fell from 0.9833 to 0.8164. Seed 0 detected 56 of 95 frauds at the validation threshold, up from 22 at 0.5, with false positives increasing from 0 to 22. FedAvg alpha 0.5 seed 2 had a small held-out F1 decrease. Each run accounted for 5.1468 decimal MB of serialized training model payload, excluding protocol overhead. [Per-run metrics](../results/minor_calibration_20261121/results.csv) · [Means and sample standard deviations](../results/minor_calibration_20261121/summary.csv)

## Artifact audit

The saved comparison plot shows means and sample standard deviations over model seeds. The seed-0 precision–recall plot marks both operating points on each model's curve, and the confusion matrices show the corresponding counts. The communication plot uses validation AUPRC during training and mean test F1 at round 30. All four figures were visually inspected. The CSV and JSON audit checked five source hashes, every paired row against its confusion totals and payload arithmetic, all six summary groups against per-run measurements, and nine histories with 30 rounds and 150 client diagnostic rows per run. The excluded prediction archives are required to independently recompute the curves and ranking metrics. [Comparison](../results/minor_calibration_20261121/comparison.png) · [Precision–recall](../results/minor_calibration_20261121/precision_recall.png) · [Confusion matrices](../results/minor_calibration_20261121/confusion_matrices.png) · [Communication](../results/minor_calibration_20261121/performance_communication.png)

## Socket reproduction

The original WAN-emulated FedAvg alpha 0.1 runs reported mean AUPRC 0.07566 and F1 0 across three seeds. They used the upstream split and a more concentrated fraud partition. A direct comparison with the controlled-study means would confound the threshold experiment with those protocol changes. Independent socket measurements are pending dataset and dependency access. [Upstream-reported snapshot](../results/minor_calibration_20261121/upstream_reported_summary.csv)

From the application directory, use the Python interpreter in an isolated virtual environment.

```sh
python -m pip install -r requirements.txt pytest
python -m pytest tests -q
python scripts/prepare_data.py --scheme dir0.1 --out data/minor_original_dir0.1
python scripts/reproduce_socket_baseline.py --parts data/minor_original_dir0.1 --rounds 30 --seed 0 --emulate-wan --output results/minor_socket_s0
```

The wrapper writes separate process logs and a run JSON to a fresh directory. Repeat for seeds 1 and 2 with distinct output directories before comparing three-seed means. Keep elapsed time and payload definitions explicit. For a loopback smoke test, omit `--emulate-wan`.

## Limitations

All three model seeds share one split and one partition per alpha. The test set has 95 frauds, and the study gives no estimate of variation across independent partitions or future time windows. Customer identifiers are unavailable and PCA features were supplied precomputed. The simulation evaluates payload and model behavior under controlled participation. Production alert costs, WAN performance and formal privacy guarantees remain open.
