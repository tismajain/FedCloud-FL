# FedCloud-FL: Complete Project Narrative and Technical Report Source

**Project:** Federated Learning Across Cloud Nodes: Non-IID Credit-Card Fraud Detection and Validation-Based Decision Thresholds  
**Repository:** [tismajain/FedCloud-FL](https://github.com/tismajain/FedCloud-FL)  
**Scope:** A consolidated, evidence-based source for preparing the Phase 1 and Phase 2 minor-specialization reports. This is a technical source document, not a replacement for the institute's prescribed Word template.  
**Evidence checkpoint:** 21 November 2026.

## Abstract

Federated learning enables a shared prediction model to be trained from data held by multiple participants without centralizing their raw training records. However, financial fraud is rare, and the prevalence of fraud can vary sharply among participating institutions. A global model trained under these non-independent and identically distributed (non-IID) conditions may assign useful relative risk scores while failing to identify many fraud cases at the conventional decision threshold of 0.5. FedCloud-FL provides a federated fraud-detection architecture built around a lightweight NumPy multilayer perceptron and multiple aggregation strategies, including FedAvg, FedProx and an asynchronous FedAsync path. The present controlled study investigates whether selecting the alert threshold on an independent validation set improves the operational behavior of the trained global model.

The study uses the ULB Credit Card Fraud Detection dataset, removes feature-identical duplicates, separates training, validation and test records, and distributes training data across five simulated clients using Dirichlet label-skew partitions. Nine 30-round training runs cover FedAvg with Dirichlet concentrations 0.1 and 0.5 and FedProx with concentration 0.1, each under three model seeds. Every final model is evaluated twice on identical held-out test scores: once at a fixed threshold of 0.5 and once at the threshold maximizing validation F1. For FedAvg at concentration 0.1, mean test F1 increases from 0.452 to 0.693 and recall from 0.298 to 0.604, while precision decreases from 0.983 to 0.816. Average precision remains 0.649 because changing a decision threshold does not alter the underlying score ranking. The findings show how validation-based operating-point selection can improve minority-class detection without retraining or additional federated model-transfer payload. The reported study is a sequential, in-process federated simulation; actual network traffic and independent reproduction of its archived real-data predictions remain outside the verified results.

## 1. Introduction and motivation

Federated learning (FL) distributes model optimization among clients that retain their local datasets. A central coordinator sends model parameters to participating clients; clients perform local training and return updated parameters for aggregation. The arrangement is relevant to financial institutions because transaction data may be confidential, institution-specific and difficult to pool in one place. FL reduces the need to exchange raw training records, although model updates alone do not guarantee formal privacy.

Two challenges arise together. First, fraud detection is an extremely imbalanced binary classification problem: only a small fraction of transactions are fraudulent. A classifier that labels nearly every transaction as legitimate can achieve high accuracy while being practically ineffective. Second, clients need not share the same fraud prevalence or transaction volume. A client holding mostly legitimate transactions can influence a sample-weighted global update more heavily than a smaller client containing important fraud examples. Consequently, both the global model's ability to rank risky transactions and the rule used to convert risk scores into fraud alerts must be evaluated.

The broader FedCloud-FL architecture supports synchronous federated averaging, proximal federated optimization, and asynchronous aggregation through networked server and client programs. The controlled minor-project investigation focuses on the last decision stage: after a federated model is trained, should the same fixed 0.5 threshold be applied to all operating conditions, or should the threshold be selected using a separate validation fold? The comparison holds model parameters and test scores fixed within each pair, so changes in F1, precision and recall can be attributed to the decision rule rather than a different trained model.

## 2. Problem statement, research questions and objectives

The problem is to evaluate fraud-alert decisions made by a federated classifier trained on severely imbalanced, non-IID client data. A global model may rank fraudulent transactions ahead of legitimate transactions but assign both groups scores well below 0.5. In that case a fixed threshold can suppress useful alerts even when the model has meaningful ranking ability.

The primary research question is: **How does selecting a decision threshold exclusively from validation scores affect held-out F1, precision and recall under different levels of client label skew, while keeping the trained model, test scores and model-transfer payload unchanged?** Supporting questions concern the differences between FedAvg and FedProx, the relationship between threshold-dependent metrics and threshold-independent ranking metrics, and the effect of client heterogeneity on the interpretation of the results.

The experimental objectives are to establish an auditable preprocessing and data-splitting protocol; train a consistent five-client federated model under specified Dirichlet partitions; compare fixed and validation-selected thresholds on exactly the same held-out predictions; report imbalance-aware metrics with variation across model seeds; account for serialized model-transfer volume; and identify limits to reproducibility and deployment claims.

## 3. Background and related work

FedAvg, introduced by McMahan et al. [1], combines locally trained models through sample-weighted aggregation. It provides a simple communication-round abstraction but can be sensitive to differences among clients' data distributions. FedProx, described by Li et al. [2], adds a proximal penalty to each local objective to discourage large deviations from the current global model. These two established algorithms form the synchronous training comparisons.

The original FedCloud-FL architecture also implements FedAsync, which aggregates updates as they arrive and discounts stale updates, together with gRPC/HTTP server-client paths, a centralized baseline, and a configurable network/straggler emulator [6]. Those system-level capabilities motivate the architecture, but the new threshold-study metrics must not be described as measurements of real cloud network behavior.

Saito and Rehmsmeier [3] explain why precision–recall analysis is especially informative for imbalanced classification. Lipton et al. [4] study the dependence of F1-optimal thresholds on classifier scores and decision populations. Accordingly, this project reports average precision (the recorded AUPRC statistic), ROC-AUC, F1, precision and recall, and selects a threshold on validation data rather than choosing it from the final test set. Threshold selection is an **operating-point optimization**, not probability calibration and not a new federated optimization algorithm.

## 4. System architecture and workflow

The project has a global coordinator, five client partitions, a shared neural-network parameter representation and an evaluation component. In the network-capable architecture, the coordinator and clients can run as separate processes; Flower 1.18 handles synchronous FedAvg/FedProx communication over gRPC, and the asynchronous implementation uses HTTP. Client-side training retains raw local records and sends parameter updates rather than transaction rows. The original repository also contains deployment instructions, a centralized baseline and a network emulator.

For the **controlled threshold study**, the same model and local-training routines are reused in a deterministic sequential Python simulation. Each of five clients trains locally for one epoch during each round, and the coordinator aggregates the updates. The study does not open real gRPC/HTTP connections or measure packet-level traffic. After the final, prespecified round, the global model produces validation and test scores. The threshold is chosen from validation scores only, frozen, and then compared with 0.5 on the same test-score vector. This distinction between network-capable architecture and in-process experimental execution is essential when reporting systems results.

The reproducible workflow is: acquire the source dataset; audit and deduplicate records; form isolated train/validation/test splits; fit preprocessing on training data; distribute only training rows among clients; initialize the shared MLP; perform federated training and aggregation; select the validation threshold from the final model; compute paired test metrics; save per-run histories, diagnostic summaries and figures; and audit the recorded artifacts.

## 5. Dataset and preprocessing

The ULB Credit Card Fraud Detection dataset contains **284,807 transactions**, including **492 fraudulent transactions**, approximately 0.173% of the original data. Its 30 input features consist of Time, Amount and 28 supplied PCA-derived components (V1–V28), with Class as the binary target [5]. The study removes **1,081 feature-identical duplicate records**, leaving **283,726 unique feature vectors**. Identical feature vectors carrying conflicting labels are treated as an error rather than silently combined.

The records are stratified with split seed 42 into three disjoint roles:

| Split | Transactions | Fraud cases | Purpose |
|---|---:|---:|---|
| Training | 170,235 | 284 | Client-side optimization and preprocessing fit |
| Validation | 56,745 | 94 | Monitoring and threshold selection |
| Test | 56,746 | 95 | Final held-out evaluation |

The Amount feature is transformed using log(1 + Amount), and feature scaling parameters are learned from the training fold alone before being applied to validation and test records. This avoids directly using validation/test statistics to fit the scaler. The split unit is a unique transaction feature vector, not a customer identity or a chronological interval. Customer-level and temporal generalization therefore remain untested. The original networked benchmark used a different data split and different client partitions; its results must not be treated as a directly comparable baseline for the new controlled study.

**Source of exact counts:** [data_manifest.json](../results/minor_calibration_20261121/data_manifest.json).

## 6. Non-IID client partitioning

The training fold is distributed across **K = 5** clients. For each class, the partitioning routine samples client proportions from a symmetric Dirichlet distribution. The concentration parameter α controls how evenly class examples are allocated: lower α generally permits stronger concentration of each class in fewer clients. The study evaluates α = 0.1 (severe skew) and α = 0.5 (moderate skew). Since class-wise proportions are drawn independently, clients can differ in both class balance and total sample count.

The split seed and partition seed are fixed at 42 for the recorded experiments. Model seeds 0, 1 and 2 vary initialization and local randomness while **sharing the same split and partition for each α**. Reported standard deviations therefore characterize variation across model seeds, not variation across independent datasets, client partitions or time periods. The upstream α = 0.1 experiment concentrated 393 of 394 training frauds in one client and is more extreme than the newly recorded α = 0.1 partition. This is why upstream and controlled-study AUPRC figures should not be interpreted as a before/after improvement.

## 7. Neural-network model and optimization

Each client uses a NumPy multilayer perceptron with layer widths **30–64–32–1**, ReLU hidden activations and a sigmoid output. The network has **4,097 trainable parameters**. The sigmoid output s_i is treated as a score for the positive (fraud) class.

The local loss is class-weighted binary cross-entropy:

[
L_k(w)=-\frac{1}{n_k}\sum_{i\in D_k}
\left[\lambda y_i\log s_i+(1-y_i)\log(1-s_i)\right],
]

where D_k is client k's training partition, n_k is its sample count, y_i is the binary target and λ = 10 is the positive-class weight. Local optimization uses Adam with learning rate 0.001, one local epoch per round and the established implementation's minibatch routine.

FedAvg aggregates the locally updated parameters using sample-count weights:

[
w^{(r+1)}=\sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}\,w_k^{(r+1)}.
]

FedProx uses the same server aggregation but adds a proximal penalty to the local objective:

[
L_k^{\mathrm{prox}}(w)=L_k(w)+\frac{\mu}{2}\lVert w-w^{(r)}\rVert_2^2,
]

with μ = 0.1 in the recorded FedProx condition. Both FedAvg and FedProx are established algorithms; the experiment assesses their behavior under a consistent evaluation protocol.

## 8. Validation-only threshold selection

A score becomes a predicted fraud alert when s_i ≥ t. The fixed operating point uses t = 0.5. The alternative selects t from the validation fold by maximizing the validation F1 score across distinct score cutoffs:

[
F_1=\frac{2TP}{2TP+FP+FN},\qquad
t^*=\arg\max_{t\in T_{\mathrm{val}}}F_1(t).
]

When multiple thresholds yield the same maximum validation F1, the implementation selects the highest such threshold. The final test set is not used to choose t*. The same final-round model and same held-out test scores are evaluated at both t = 0.5 and t = t*. This paired design isolates the effect of changing the decision rule.

Precision = TP/(TP+FP) measures the fraction of predicted alerts that are true frauds; recall = TP/(TP+FN) measures the fraction of actual frauds detected. AUPRC, recorded as average precision, summarizes ranking performance across score cutoffs. ROC-AUC measures positive–negative score ordering. Because the test scores are unchanged between the two operating points, **AUPRC and ROC-AUC must remain unchanged**. A threshold can alter alert counts, F1, precision and recall but cannot improve the ranking of transactions.

## 9. Experimental setup and reproducibility

The recorded study consists of **nine final trained models** and **18 paired test rows**:

| Condition | Aggregation | Dirichlet α | Model seeds |
|---|---|---:|---|
| A | FedAvg | 0.1 | 0, 1, 2 |
| B | FedAvg | 0.5 | 0, 1, 2 |
| C | FedProx (μ = 0.1) | 0.1 | 0, 1, 2 |

Each run uses five fully participating clients, 30 communication-round equivalents, one local epoch, learning rate 0.001 and positive loss weight 10. The final checkpoint is round 30 for all comparisons. The saved run configuration records the Python/package versions and source hashes used for the archived experiments: [config.json](../results/minor_calibration_20261121/config.json).

The study counts a serialized parameter payload of **17,156 bytes per model transfer**. Five clients each receive and return one payload per round; therefore the simulated total is 2 × 5 × 30 × 17,156 = **5,146,800 bytes**, or **5.1468 decimal MB** per run. This is model-payload accounting, **not measured network traffic**. It excludes protocol headers, control messages and actual network conditions. Threshold selection is performed at the coordinator after training, so it does not require additional federated model updates.

The raw dataset, generated client archives and prediction NPZ files are excluded from the published repository to avoid redistributing large artifacts. Saved CSV, JSON and PNG artifacts document the reported runs. Recomputing the archived ranking metrics independently requires regenerated predictions or the original excluded prediction archives.

## 10. Results and interpretation

The following values are **means across three model seeds**, using one fixed split and one fixed partition per α. They come from the archived [summary.csv](../results/minor_calibration_20261121/summary.csv).

| Strategy | α | Threshold | AUPRC | F1 | Precision | Recall | ROC-AUC |
|---|---:|---|---:|---:|---:|---:|---:|
| FedAvg | 0.1 | Fixed 0.5 | 0.6490 | 0.4519 | 0.9833 | 0.2982 | 0.8909 |
| FedAvg | 0.1 | Validation F1 | 0.6490 | 0.6931 | 0.8164 | 0.6035 | 0.8909 |
| FedAvg | 0.5 | Fixed 0.5 | 0.7713 | 0.7572 | 0.8135 | 0.7123 | 0.9415 |
| FedAvg | 0.5 | Validation F1 | 0.7713 | 0.7745 | 0.8740 | 0.6982 | 0.9415 |
| FedProx | 0.1 | Fixed 0.5 | 0.6996 | 0.6617 | 0.8635 | 0.5368 | 0.9327 |
| FedProx | 0.1 | Validation F1 | 0.6996 | 0.7441 | 0.8327 | 0.6737 | 0.9327 |

### 10.1 Severe skew: FedAvg, α = 0.1

The largest mean improvement occurs under severe skew. F1 rises from **0.4519 to 0.6931**, an absolute gain of **0.2412** and a relative gain of approximately **53.4%**. Recall rises from **0.2982 to 0.6035**, roughly doubling the fraction of fraud cases detected. Precision falls from **0.9833 to 0.8164**, demonstrating the trade-off: more frauds are caught at the expense of more false alerts. The seed-0 run identifies **56 of 95** test frauds at the selected threshold versus **22 of 95** at 0.5, while false positives rise from **0 to 22**. The mean selected threshold is approximately **3.79 × 10⁻⁵**, indicating that absolute model scores under this condition are much lower than the conventional 0.5 cutoff. This threshold is empirical and should not be interpreted as a calibrated probability of fraud.

### 10.2 Moderate skew: FedAvg, α = 0.5

FedAvg under moderate skew achieves higher ranking quality, with mean AUPRC **0.7713**. Threshold selection changes mean F1 from **0.7572 to 0.7745**, a smaller gain than in the severe-skew case. Precision increases from **0.8135 to 0.8740** while recall decreases slightly from **0.7123 to 0.6982**. Thus, threshold optimization does not always mean lowering the threshold or increasing recall; it selects the operating point that maximizes F1 on validation data. One model seed shows a slight held-out F1 decline, emphasizing that validation selection is not guaranteed to improve every unseen test sample.

### 10.3 Severe skew: FedProx, α = 0.1

FedProx achieves mean AUPRC **0.6996**. Mean F1 improves from **0.6617 to 0.7441** after validation-based threshold selection, while recall increases from **0.5368 to 0.6737** and precision decreases from **0.8635 to 0.8327**. This supports the observation that the operating threshold matters for both federated optimization strategies, although the size of the effect depends on the training condition and score distribution.

### 10.4 Ranking metrics, communication and uncertainty

In every paired comparison, AUPRC and ROC-AUC remain identical because the scores and labels are identical. The result is **better decision performance at a selected threshold**, not improved ranking performance or a more accurate probability model. The serialized training payload remains **5.1468 MB** for each recorded run because selecting a threshold after training does not change the federated training procedure. Error bars in the saved comparison figure represent **sample standard deviation over three model seeds**. They do not establish statistical significance across independent datasets.

**Figures for the report:**

- [Figure: AUPRC, F1 and recall comparison](../results/minor_calibration_20261121/comparison.png)
- [Figure: Precision–recall curves and selected operating points](../results/minor_calibration_20261121/precision_recall.png)
- [Figure: Test confusion matrices](../results/minor_calibration_20261121/confusion_matrices.png)
- [Figure: Performance and serialized model payload](../results/minor_calibration_20261121/performance_communication.png)

The [per-run results](../results/minor_calibration_20261121/results.csv), [history](../results/minor_calibration_20261121/history.json) and [experiment record](EXPERIMENTS.md) provide the traceable basis for tables and plots.

## 11. Verification, testing and evidence boundaries

The project includes unit and scientific-contract tests covering threshold selection, ties and invalid inputs, data preprocessing, leakage boundaries, deterministic training, aggregation parity and communication arithmetic. On **21 November 2026**, the project was cloned to a clean Windows Python 3.12.10 virtual environment, dependencies installed successfully, and the user ran:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -rs
```

The reported output was **13 passed, 4 warnings in 24.25 seconds**, with **zero skips and zero failures**. The four warnings concerned deprecations in dependencies, not failing tests. This local verification resolves the previously blocked clean-environment test-suite check, including the Flower aggregation parity test. This result is user-supplied terminal evidence, not a CI run committed to GitHub.

The archived experiment audit also checked source hashes, paired metric/confusion arithmetic, 18 test rows, nine 30-round histories and six summary groups. These checks establish internal consistency of the saved records. They **do not** independently recompute the archived AUPRC and ROC-AUC from the excluded original prediction arrays. A fresh nine-run real-dataset reproduction and original networked socket baseline remain unverified. The user's local machine does not currently store the full real dataset, and no new dataset download is assumed or required for using this report source.

A deterministic **synthetic-data demonstration** is available through `scripts/run_threshold_demo.py`; it can exercise the pipeline without downloading the full ULB dataset. Synthetic demo metrics must not be presented as the real-data study results. For a local run from `fedcloud-fl`, after installing requirements, use `python scripts/run_threshold_demo.py`. See [DEMONSTRATION.md](DEMONSTRATION.md) for the procedure and expected artifacts.

## 12. Discussion

The findings separate two questions often conflated in fraud-detection reporting: whether a model can rank fraudulent transactions above legitimate ones, and whether a chosen threshold produces useful alerts. The saved severe-skew FedAvg model has mean AUPRC 0.649 but a fixed-threshold recall of only 0.298. The paired experiment demonstrates that changing the decision boundary can reveal more of the model's available detection capability without changing its ranking or retraining the model.

This distinction is operationally relevant because different fraud-review teams may have different alert capacities and false-positive costs. Maximizing F1 provides a reproducible research objective, but it is not automatically the best production policy. A deployed system would need threshold selection informed by review capacity, monetary costs, prevalence drift and temporal validation. Threshold values can also be unstable when the validation fold contains relatively few positives.

The study's communication result is similarly specific: it establishes unchanged **serialized model payload** under the two post-training operating points. It does not establish unchanged real-world packet volume, latency or cloud cost. The repository contains code for broader socket-based benchmarks, but the controlled threshold study itself is an in-process simulation.

## 13. Limitations and future work

The evidence comes from one public dataset, one stratified split and one fixed client partition per concentration parameter. The test set contains 95 fraud cases, and the validation set contains 94. Three model seeds provide limited uncertainty characterization and do not represent repeated data collection or independent client partitions. The absence of customer identifiers prevents customer-level separation; the random split does not assess future-period fraud drift. The PCA representation is provided rather than learned in this pipeline. The study assumes full participation by five clients and does not measure dropout, network delays, actual cross-cloud traffic or production review costs. No differential privacy or secure aggregation guarantee is established by the reported results.

Future work can evaluate multiple partition seeds and client counts, chronological holdouts, fraud-cost-aware threshold objectives, calibration of score probabilities as a separate problem, class-aware training or aggregation methods that genuinely improve AUPRC, and actual distributed network runs with measured latency and payload overhead. These are proposed extensions, **not completed results**.

## 14. Conclusion

FedCloud-FL provides a federated-learning setting for studying credit-card fraud under heterogeneous client distributions. Its controlled five-client experiment shows that validation-only decision-threshold selection can substantially improve held-out fraud-detection F1 and recall under severe label skew, even when the trained model and its ranking metrics are unchanged. The strongest recorded FedAvg condition increases mean F1 from 0.452 to 0.693 and recall from 0.298 to 0.604, while reducing precision from 0.983 to 0.816. FedProx exhibits a smaller but meaningful F1 improvement under the same skew concentration. These results support treating decision thresholds as explicit, validated operating choices rather than assuming that 0.5 is universally appropriate. The work is supported by archived experiments and a passing local test suite, subject to the stated reproduction and deployment limitations.

## 15. How to turn this source into the prescribed academic report

Use the institute's existing Word template **without replacing its formatting or cover pages**. Transfer the prose into the template's required chapter headings, preserve separate preliminary pages for certificate, declaration, acknowledgement, abstract and contents as specified by the template, and start each major chapter on a new page. Use formal paragraph prose, numbered figures and tables, equations with symbols defined, and numbered references. Do not insert checkmark symbols, status badges or software-development task lists into the academic report. Where the template requests details not present in the repository (student registration number, guide name, approval signatures), leave them for authorized completion. The technical story should read continuously across architecture, methodology and results; scholarly citations should still identify established methods and upstream source material.

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, “Communication-efficient learning of deep networks from decentralized data,” *Proc. AISTATS*, PMLR, vol. 54, pp. 1273–1282, 2017. https://proceedings.mlr.press/v54/mcmahan17a.html

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, “Federated optimization in heterogeneous networks,” *Proc. MLSys*, vol. 2, 2020. https://arxiv.org/abs/1812.06127

[3] T. Saito and M. Rehmsmeier, “The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets,” *PLOS ONE*, vol. 10, no. 3, e0118432, 2015. https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432

[4] Z. C. Lipton, C. Elkan, and B. Naryanaswamy, “Optimal thresholding of classifiers to maximize F1 measure,” *ECML PKDD*, 2014. https://arxiv.org/abs/1402.1892

[5] ULB Machine Learning Group, “Credit Card Fraud Detection,” dataset. https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

[6] P. Kasliwal, N. Neelesh, H. Rebala, and D. Saha, “FedCloud-FL: Federated Learning Across Cloud Nodes,” original implementation and reported network experiments, source revision `3ba5dd0c`. https://github.com/deepankitasaha/cc

## Primary project evidence and code locations

| Material | Repository path |
|---|---|
| Main model and local optimization | `fedcloud/model.py` |
| Client partitioning and original dataset pipeline | `fedcloud/data.py` |
| Deduplicated study preprocessing | `fedcloud/study_data.py` |
| Validation-only threshold selection | `fedcloud/calibration.py` |
| Synchronous server/client | `fedcloud/sync_server.py`, `fedcloud/sync_client.py` |
| Asynchronous server/client | `fedcloud/async_server.py`, `fedcloud/async_client.py` |
| Nine-run study driver | `scripts/run_threshold_study.py` |
| Synthetic-data demo | `scripts/run_threshold_demo.py` |
| Ranking verifier | `scripts/verify_study_rankings.py` |
| Tests | `tests/test_calibration.py`, `tests/test_smoke.py` |
| Configuration and provenance | `results/minor_calibration_20261121/config.json`, `data_manifest.json` |
| Saved measurements | `results/minor_calibration_20261121/results.csv`, `summary.csv`, `history.json` |
| Full experiment notes | `docs/EXPERIMENTS.md` |
| Verification handoff | `docs/PROGRESS.md` |
| Attribution details | `docs/CONTRIBUTIONS.md` |

**Source attribution note:** The project narrative deliberately presents the architecture and experiment as a coherent technical account. FedAvg, FedProx, the original FedCloud-FL code and its previously reported network results are established prior work; the controlled validation-threshold study, preprocessing protocol, paired analysis and associated verification are additions documented in this repository. See [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for explicit provenance.
