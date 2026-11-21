# Decision thresholds for non-IID federated fraud detection

## Abstract

Uneven fraud prevalence across federated clients can yield a shared model whose rankings are informative while a threshold of 0.5 misses many frauds. We evaluated validation-only threshold selection in nine fully participating, five-client simulations using the ULB Credit Card Fraud Detection dataset. Each fitted model produced one held-out score vector evaluated with a fixed threshold of 0.5 and a threshold maximizing validation F1. Under Dirichlet alpha 0.1, mean test F1 increased from 0.452 to 0.693 for FedAvg and from 0.662 to 0.744 for FedProx across three initialization seeds. FedAvg recall increased from 0.298 to 0.604 as precision decreased from 0.983 to 0.816. Average precision stayed at 0.649 and 0.700, respectively, across the two operating points. Each run accounted for 5.1468 MB of simulated training model payload. These measurements describe threshold behavior on a single fixed data split and partition per alpha.

## Problem statement

Fraud is rare, and different client partitions can contain very different numbers of fraudulent transactions. A global model trained on these partitions still needs an operating rule that turns scores into alerts. A fixed cutoff may miss high-ranking fraud cases with low absolute scores. The research question is how much validation-only threshold selection changes held-out fraud detection when the trained model, test scores and training payload are identical within each comparison.

## Objectives

1. Construct a reproducible label-skew study with isolated training, validation and test roles.
2. Measure paired changes in F1, precision and recall across fixed and validation-selected thresholds.
3. Track average precision, ROC AUC and model payload so readers can interpret decision changes alongside ranking and communication.
4. Preserve the data provenance, study configuration and per-run diagnostics required for reproduction.

## Literature review

McMahan et al. introduced FedAvg, which aggregates multiple local optimization steps through sample-weighted averaging. Their work motivates the use of local training and communication rounds as experimental units. Li et al. proposed FedProx for heterogeneous federations. It adds a proximal penalty that discourages clients from moving too far from the current server model. Here FedProx provides a second training condition under severe skew. The paired threshold comparison holds the final model fixed within both algorithms. [1](#references) [2](#references)

Saito and Rehmsmeier show why precision–recall analysis is informative under severe imbalance. Precision exposes how many alerts correspond to positives, including cases where even a low false-positive rate creates many alerts. Lipton et al. analyze the dependence of F1-optimal thresholds on scores and the decision population. We choose thresholds empirically on held-out validation scores and report precision and recall alongside F1. This criterion models a specific research objective. Production fraud review would require alert costs and capacity constraints. [3](#references) [4](#references)

## Dataset and preprocessing

The ULB dataset contains 284,807 labeled transactions, including 492 frauds (0.173%). Its columns include elapsed Time, transaction Amount, 28 supplied PCA components and the binary Class label. The study removed 1,081 feature-identical duplicates before splitting. Conflicting labels for identical feature vectors cause preprocessing to fail. Amount undergoes a log1p transform, then the scaler fits the training fold and transforms all folds with the same parameters. The split is stratified and fixed at seed 42. [Dataset and split manifest](../results/minor_calibration_20261121/data_manifest.json)

| Fold | Transactions | Frauds | Role |
|---|---:|---:|---|
| Training | 170,235 | 284 | Scaling and local model updates |
| Validation | 56,745 | 94 | Learning curves and threshold selection |
| Test | 56,746 | 95 | Final paired evaluation |

## Mathematical methodology

For model parameters $w$, the 30–64–32–1 MLP assigns transaction $x_i$ a sigmoid score $s_i$. Each client $k$ with $n_k$ training rows minimizes weighted binary cross-entropy using Adam. The positive weight $\lambda=10$ increases the loss contribution of a fraud example.

$$
L_k(w)=-\frac{1}{n_k}\sum_{i\in D_k}[\lambda y_i\log(s_i)+(1-y_i)\log(1-s_i)].
$$

At round $r$, FedProx adds $\frac{\mu}{2}\lVert w-w^r\rVert_2^2$ to the client's training objective. FedAvg uses $\mu=0$. The server updates its model using participating clients' trained weights $w_k^{r+1}$.

$$
w^{r+1}=\sum_{k=1}^{5}\frac{n_k}{\sum_j n_j}w_k^{r+1}.
$$

For a threshold $t$, the alert rule is $\hat y_i=\mathbf{1}[s_i\geq t]$. On the validation set, the selection routine evaluates distinct score thresholds and selects the highest threshold among those with maximal $F_1=2TP/(2TP+FP+FN)$. Test precision is $TP/(TP+FP)$ and test recall is $TP/(TP+FN)$. Average precision sums precision weighted by successive recall increments over distinct score cutoffs. ROC AUC measures the fraction of correctly ordered positive–negative pairs, giving half credit to ties. Both ranking metrics depend on the scores and labels and are invariant to the choice of alert cutoff.

## Experimental design

The study drew class-wise client shares from symmetric Dirichlet distributions, then assigned the training rows to five clients. Smaller alpha values favor more concentrated shares. The conditions were FedAvg at alpha 0.1 and 0.5, and FedProx at alpha 0.1 with $\mu=0.1$. All five clients participated in each of 30 rounds, with one local epoch and learning rate 0.001. Partition seed 42 was fixed. Model seeds 0, 1 and 2 yielded nine models. Validation metrics monitored training, and all evaluations used the prespecified round-30 checkpoint. Each model used the same test scores under the two decision rules. [Recorded configuration](../results/minor_calibration_20261121/config.json)

Each client received and returned a serialized parameter payload of 17,156 bytes per round. The total was $2\times5\times30\times17{,}156=5{,}146{,}800$ bytes. This accounts for the simulated training transfers. Test threshold selection occurs server-side. Actual packet volume, network latency and cloud operating costs require a networked measurement.

## Results

The table reports saved means across three model seeds on each fixed partition. Average precision is the study's AUPRC statistic. [Per-run results](../results/minor_calibration_20261121/results.csv) · [Means and sample standard deviations](../results/minor_calibration_20261121/summary.csv)

| Strategy | Alpha | Threshold | Average precision | F1 | Precision | Recall | ROC AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| FedAvg | 0.1 | 0.5 | 0.649 | 0.452 | 0.983 | 0.298 | 0.891 |
| FedAvg | 0.1 | Validation F1 | 0.649 | 0.693 | 0.816 | 0.604 | 0.891 |
| FedAvg | 0.5 | 0.5 | 0.771 | 0.757 | 0.813 | 0.712 | 0.942 |
| FedAvg | 0.5 | Validation F1 | 0.771 | 0.775 | 0.874 | 0.698 | 0.942 |
| FedProx | 0.1 | 0.5 | 0.700 | 0.662 | 0.863 | 0.537 | 0.933 |
| FedProx | 0.1 | Validation F1 | 0.700 | 0.744 | 0.833 | 0.674 | 0.933 |

The largest mean F1 increase was 0.241 for FedAvg at alpha 0.1. Its seed-0 run detected 56 of 95 test frauds with the validation threshold and 22 at 0.5. False positives increased from 0 to 22. The smaller mean gain at alpha 0.5 included one seed where F1 decreased from 0.790 to 0.788. Both ranking metrics were identical within all nine pairs. [Confusion matrices](../results/minor_calibration_20261121/confusion_matrices.png) · [Precision–recall curves](../results/minor_calibration_20261121/precision_recall.png)

The upstream socket-based FedAvg alpha 0.1 experiment reports mean average precision of 0.07566 and mean F1 of 0. Its original split and concentration of client fraud labels differ from the controlled study. Independent socket reproduction and verification on regenerated local prediction archives remain pending. [Experiment record](EXPERIMENTS.md) · [Verification status](PROGRESS.md)

## Limitations

The three model seeds share one split and one partition per alpha. Sample standard deviations describe variability across these initializations. Variation across partitions and future periods remains unknown. The test set contains 95 frauds, and the validation set contains 94, so individual cases can affect both threshold selection and the observed test outcome. Customer identifiers are unavailable, and a stratified random split provides no temporal validation. PCA features were supplied precomputed. The simulation leaves WAN behavior and client dropout unmeasured. Formal privacy guarantees and alert-handling costs are outside the evidence collected here.

## Conclusion

With the saved alpha 0.1 partitions, validation-only threshold selection improved mean F1 for both algorithms without additional training transfers. Its most pronounced effect was increased recall for FedAvg alongside lower precision. AUPRC and ROC AUC stayed constant within each model, so further improvements in ranking require changes to training or data. Future evaluation should vary partitions, hold out later periods and quantify uncertainty and alert costs.

## Original contribution

This work contributes the deduplicated evaluation protocol, validation-only threshold selection, paired experiment harness, diagnostics, ranking verification and reproducibility materials. Pranav Kasliwal, Nandini Neelesh, Harshini Rebala and Deepankita Saha supplied the upstream implementation, including the NumPy MLP, partitioning, networked training and original results. McMahan et al. and Li et al. introduced the federated algorithms evaluated here. [Contribution boundaries](CONTRIBUTIONS.md)

## References

1. McMahan, B., Moore, E., Ramage, D., Hampson, S., and Agüera y Arcas, B. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data*. AISTATS, PMLR 54, 1273–1282. [Paper](https://proceedings.mlr.press/v54/mcmahan17a.html)
2. Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., and Smith, V. (2020). *Federated Optimization in Heterogeneous Networks*. MLSys. [Paper](https://arxiv.org/abs/1812.06127)
3. Saito, T., and Rehmsmeier, M. (2015). *The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets*. PLOS ONE, 10(3), e0118432. [Paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432)
4. Lipton, Z. C., Elkan, C., and Naryanaswamy, B. (2014). *Optimal Thresholding of Classifiers to Maximize F1 Measure*. ECML PKDD. [Author manuscript](https://arxiv.org/abs/1402.1892)
5. ULB Machine Learning Group. *Credit Card Fraud Detection*. [Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
6. Kasliwal, P., Neelesh, N., Rebala, H., and Saha, D. *FedCloud-FL*, revision 3ba5dd0. [Upstream source](https://github.com/deepankitasaha/cc)
