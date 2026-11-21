# Presentation outline

1. **Problem.** Explain severe fraud imbalance and uneven fraud counts across five simulated clients. Introduce average precision and the alerting threshold.
2. **Controlled design.** Show deduplication, training-only scaling, the fixed split and the validation-to-test decision flow. Identify the FedAvg and FedProx conditions.
3. **Main result.** Use the [comparison plot](../results/minor_calibration_20261121/comparison.png) to show fixed versus tuned F1, recall and AUPRC across three model seeds on the same split.
4. **Alert trade-off.** Use the [confusion matrices](../results/minor_calibration_20261121/confusion_matrices.png) to show that FedAvg alpha 0.1 seed 0 detected 34 additional frauds with 22 additional false positives. Show the two operating points on the [precision–recall curve](../results/minor_calibration_20261121/precision_recall.png).
5. **Communication.** Use the [payload figure](../results/minor_calibration_20261121/performance_communication.png) to show the shared 5.1468 MB training budget and explain what serialized model payload covers.
6. **Limits and next steps.** Cover the single partition per alpha, the pending socket reproduction, and the need for time-based validation and operational cost estimates. Credit the upstream team and established algorithms. [Report and references](MINOR_PROJECT_REPORT.md)
