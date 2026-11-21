# Attribution and contribution boundaries

The source is [deepankitasaha/cc](https://github.com/deepankitasaha/cc), FedCloud-FL revision `3ba5dd0c0950e665f2bf6ac88d57e4c0c547fe2e`. Upstream README credits Pranav Kasliwal, Nandini Neelesh, Harshini Rebala and Deepankita Saha. Preserve those credits.

Upstream supplied the NumPy MLP, weighted BCE and Adam, FedAvg and FedProx Flower processes, HTTP FedAsync, network emulation, dataset downloader, label-skew partitioning, serialization, centralized baseline, smoke tests, and original reported results/paper. These are not new contributions.

Our additions are the validation-only decision-threshold selection module, deduplicated train/validation/test methodology, controlled reproducible experiment harness, explicit separation of threshold versus ranking effects, client diagnostics, plots, scientific-contract tests, safe socket reproduction wrapper, and minor-project documentation. Threshold maximization, FedAvg and FedProx are established methods; the original contribution is their carefully tested application and evaluation here, not invention of a new FL algorithm.

The new publishable branch is a clean snapshot because upstream Git history contains dataset archives. Original history is retained locally and remains accessible at the upstream URL. No dataset archives or environment files are redistributed. Existing upstream results are clearly distinguished from new measured results under `results/minor_calibration_20261121/`.

The continuation added wrapper input validation, an upstream WAN-emulation option, and a documented audit of the original study artifacts. Independent socket verification remains pending. [Experiments](EXPERIMENTS.md)
