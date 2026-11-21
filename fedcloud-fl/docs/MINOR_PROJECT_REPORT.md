# Decision thresholds for non-IID federated fraud detection

## Research question

How much does validation-only threshold selection change test-set fraud detection under label skew when each model's scores and the training payload are held fixed?

## Method

The controlled experiment reused the upstream NumPy MLP and compared FedAvg with FedProx under Dirichlet label skew. Feature-identical duplicates were removed before the fixed stratified 60/20/20 split. Five simulated clients participated for 30 rounds. Each final model was scored twice on the same held-out test set, once with a 0.5 threshold and once with a threshold selected to maximize validation F1. Average precision (reported as AUPRC) evaluates score ranking. F1, precision and recall describe alert behavior at an operating threshold. [Experiment protocol and artifacts](EXPERIMENTS.md)

## Findings

At alpha 0.1, validation threshold selection increased mean F1 from 0.452 to 0.693 for FedAvg and from 0.662 to 0.744 for FedProx across three model seeds. For FedAvg, the gain came with higher recall and lower precision. AUPRC stayed at 0.649 for FedAvg and 0.700 for FedProx across thresholds. The observed AUPRC difference between strategies describes this partition and these runs. Each run used 5.1468 MB of simulated model payload. [Measured summary](../results/minor_calibration_20261121/summary.csv)

## Interpretation

Validation-based threshold selection can recover detections from a model whose scores concentrate below 0.5. The unchanged AUPRC shows the remaining ranking problem. One fixed partition per alpha and 95 test frauds limit generalization. The test concerns transaction feature vectors, with customer-level independence and performance under time drift still open. The upstream socket baseline remains pending independent reproduction, and the controlled protocol differs from the upstream protocol. Before academic submission, assess independent partitions, chronological holdouts, uncertainty and alert handling costs. [Progress and verification status](PROGRESS.md)

## References and attribution

- McMahan et al. (2017), *Communication-Efficient Learning of Deep Networks from Decentralized Data*. [AISTATS proceedings](https://proceedings.mlr.press/v54/mcmahan17a.html)
- Li et al. (2020), *Federated Optimization in Heterogeneous Networks*. [Author manuscript](https://arxiv.org/abs/1812.06127)
- ULB Machine Learning Group, *Credit Card Fraud Detection*. [Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Kasliwal, Neelesh, Rebala and Saha, original FedCloud-FL implementation. [Contribution boundaries](CONTRIBUTIONS.md)
