# Threshold study demonstration

## Walk through the saved study

1. Open the [data manifest](../results/minor_calibration_20261121/data_manifest.json) and identify the training, validation and test populations. Explain that training uses five label-skewed client partitions while the validation fold selects one threshold per final model.
2. Open the [comparison figure](../results/minor_calibration_20261121/comparison.png) and show the paired F1 changes. Explain that the AP values use the same test scores for both rules.
3. Open the [confusion matrices](../results/minor_calibration_20261121/confusion_matrices.png). In FedAvg alpha 0.1 seed 0, changing the threshold increased true positives from 22 to 56 and false positives from 0 to 22. Discuss the corresponding alert workload.
4. Open the [payload figure](../results/minor_calibration_20261121/performance_communication.png) and identify 5.1468 MB of serialized training payload per run. Describe the controlled study as a sequential simulation with five clients per round.

These files are sufficient for the walkthrough without package installation or a dataset download.

## Reproduce all nine runs

Run the commands from `fedcloud-fl` using Python 3.12. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv-study
$studyPython = ".venv-study/Scripts/python.exe"
& $studyPython -m pip install -r requirements.txt pytest
& $studyPython -m pip check
& $studyPython -m pytest tests -q -rs
```

On Linux or macOS, use `python3.12 -m venv .venv-study` and invoke `.venv-study/bin/python` in the commands. If PyPI access is unavailable and a trusted platform-compatible wheelhouse is available locally, install with `--no-index --find-links <wheelhouse>`. After a passing clean run, capture `python -m pip freeze` as the platform-specific environment record. The saved run configuration lists recorded versions for comparison.

Obtain the ULB dataset from its [publisher listing](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place the extracted CSV at `data/creditcard.csv`. Compare its SHA-256 with the [saved manifest](../results/minor_calibration_20261121/data_manifest.json) before interpreting reproducibility differences. In PowerShell, use `Get-FileHash data/creditcard.csv -Algorithm SHA256`.

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
& $studyPython scripts/run_threshold_study.py --csv data/creditcard.csv --output results/minor_regenerated --rounds 30 --epochs 1 --lr 0.001 --pos-weight 10 --clients 5 --seeds 0 1 2 --split-seed 42 --partition-seed 42 --schemes dir0.1 dir0.5 --include-fedprox
& $studyPython scripts/verify_study_rankings.py results/minor_regenerated
```

Use a fresh output directory every time. A complete run contains nine prediction archives and 18 test rows. A successful verification reports nine runs and 18 measurements. Compare the new and saved configurations and manifests, then join the two CSVs on `run` and `method`. Compare threshold, AP, ROC AUC, F1, precision, recall, test counts and payload for each pair. Report any differences with both values and check runtime and package versions. Treat timing separately because it depends on the machine.

Keep datasets, generated prediction and model archives, virtual environments and credentials local. See the separate [upstream socket protocol](EXPERIMENTS.md#socket-reproduction) for the Flower baseline.
