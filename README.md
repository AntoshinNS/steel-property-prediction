# SteelBench Public Research Supplement — v1.0

Computational supplement to **Explainable Machine Learning for Steel Mechanical Property Prediction: Generalization and Attribution Value Across Unseen Grades and Families**.

Authors: Valery E. Pyatetsky, Nikita S. Antoshin, Maksim S. Vinokurov, Alexander A. Zinin, Alexander V. Zakorko, Ksenia P. Kirchu.

Affiliation: Department of Business Informatics and Production Management Systems, National University of Science and Technology MISIS, Moscow, Russia.

This package contains the input and processed data, experiment configuration, analysis software, saved predictions and attributions, preprocessing controls, figures, and verification evidence. It provides the computational materials accompanying the manuscript.

## Contents

| Path | Contents |
|---|---|
| `work/dataset/steelbench_core_open.csv` | Supplied open CSV: 1,360 records and 25 columns |
| `work/dataset/DATASET_CARD.md`, `croissant.json`, `LICENSE` | Original upstream metadata and data-license notice, preserved as supplied |
| `work/real_run/steelbench_core_open_analysis.csv` | Analysis CSV: the same 1,360 records plus `analysis_unit` |
| `work/real_run/DATASET_CARD.md` | Description and statistics of the exact packaged analysis data |
| `work/real_run/steelbench_config_real_fast.json` | Fixed configuration used for the main results |
| `work/real_run/results_uts_fast/` | Predictions, internal OOF diagnostics, attribution arrays, fit/split manifests and summary metrics |
| `work/code/steelbench_research/` | Experiment, saved-output checker, eight unit tests and pinned dependencies |
| `work/build_analysis.py` | Reconstruct the analysis CSV from the input CSV |
| `work/revision_analysis.py` | Rerun the predictive preprocessing controls |
| `work/revision_checks.py` | Check data, split boundaries, feature overlap and saved metrics |
| `work/plot_revision_figures.py` | Recreate Figs. 2–4 from the saved results |
| `work/revision/` | Control-experiment outputs, split statistics, family/source summaries, figures and verification tables |
| `checks/` | Earlier verification logs |
| `checks/latest/` | Latest code-verification report, logs and synthetic test inputs |
| `LICENSE`, `LICENSING.md`, `CITATION.cff` | Software license, component-specific licensing and citation metadata |
| `SHA256SUMS.txt` | SHA-256 checksum of every other file in the package |

## Data and source

The packaged analysis data contain **1,360 records, 562 supplied grade identifiers, 17 family labels, three source tiers and 683 derived split units**. All records have observed ultimate tensile strength (UTS), in MPa. Missing predictors are retained in the CSV and handled within each training fit.

The original upstream dataset card describes a larger 1,636-record release. Its counts refer to that upstream description; the actual files in this package are described in `work/real_run/DATASET_CARD.md`. The supplied raw CSV and the processed CSV both contain 1,360 records. Processing adds the grouping column `analysis_unit`.

The upstream release cited by the manuscript is **SteelBench v1.0**, DOI [10.5281/zenodo.18530558](https://doi.org/10.5281/zenodo.18530558). The public reference source for the EMK handbook/specification subset is the encyclopedia of the European Metallurgical Company (EMK24): [emk24.uz/wiki](https://emk24.uz/wiki/) and [emk24.ru/wiki](https://emk24.ru/wiki/). The source-tier labels and supplied data values are preserved.

## Environment

Use Python 3.12 and the pinned packages in `work/code/steelbench_research/requirements.txt`. The latest verification used Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, SciPy 1.17.0, scikit-learn 1.8.0, matplotlib 3.10.8 and joblib 1.5.3. The saved primary run records Python 3.12.13 in its environment file.

## Reproduce and verify

Run all commands from this extracted package root. Creating a virtual environment is recommended.

```sh
python -m venv .venv
# Linux/macOS:
. .venv/bin/activate
# On Windows use: .venv\Scripts\activate
python -m pip install -r work/code/steelbench_research/requirements.txt

# Verify the provided outputs and run the software tests.
python work/code/steelbench_research/verify_saved.py work/real_run/results_uts_fast
python -m unittest discover -s work/code/steelbench_research -p test_experiment.py
python work/revision_checks.py

# Reconstruct the processed CSV into a fresh file.
python work/build_analysis.py --raw work/dataset/steelbench_core_open.csv --out rebuilt_analysis.csv

# Rerun the predictive preprocessing controls and numerical charts.
python work/revision_analysis.py
python work/revision_checks.py
python work/plot_revision_figures.py

# Refit the main nested experiment into a fresh directory.
python work/code/steelbench_research/experiment.py run --data work/real_run/steelbench_core_open_analysis.csv --config work/real_run/steelbench_config_real_fast.json --out new_full_run
python work/code/steelbench_research/verify_saved.py new_full_run

# Run the small synthetic end-to-end software check.
python work/code/steelbench_research/experiment.py run --data checks/latest/smoke/SYNTHETIC_NOT_STEEL.csv --config checks/latest/smoke/all_protocols_config.json --out new_smoke_run
python work/code/steelbench_research/verify_saved.py new_smoke_run
```

The reconstruction script requires a new output filename. The control and plotting scripts write to `work/revision/`. Run them in a working copy when retaining the exact archived outputs.

## Evaluation and interpretation

The four protocols hold out random grouped units, grades, steel families or source tiers. Each outer fold constructs its diagnostic models from inner out-of-fold (OOF) prediction errors. Imputation and scaling are fitted on the applicable training data. The diagnostic rule `g0` uses training-data distance and ensemble dispersion; `g1` additionally uses attribution spread.

The fixed attribution budget is five fitted pipelines, up to 16 background records and eight permutation pairs. The code uses a permutation estimator of empirical interventional Shapley values with reversed permutations. The normalized area under the error–retention curve (AURC) is evaluated over retained fractions 0.5–1.0. Positive `AURC(g0) − AURC(g1)` favors the augmented rule.

The mean paired family effect is **4.2774 MPa**, with conditional 95% family-bootstrap interval **[−0.4787, 9.8448] MPa**. The comparison is conditional on the observed families, fitted procedure and stated computational budget. Source/family composition, feature overlap and preprocessing controls are documented in the supplied audit tables.

## Verification evidence

The latest audit passed eight unit tests and checks of 117 outer/internal split boundaries across 30 outer folds. It reconstructed the processed CSV exactly, reproduced all 5,610 predictive-control predictions exactly, restored the diagnostic scores for all 30 outer folds to machine precision, and reproduced the three numerical PNG charts pixel for pixel.

On SteelBench, preprocessing controls were refitted and primary outputs were checked using the saved outer and inner OOF results. The full nested workflow was additionally executed on the clearly labeled synthetic software-test data under all four protocols. See `checks/latest/CODE_VERIFICATION.md` for the scope and evidence.

## License and citation

Analysis software is provided under the **MIT License** in `LICENSE`. The supplied dataset retains its upstream **CC BY 4.0** notice in `work/dataset/LICENSE`. See `LICENSING.md` for the scope of each license and attribution requirements. Citation metadata for this supplement are provided in `CITATION.cff`; cite the upstream dataset separately when using its data.

## Input integrity

- Input CSV SHA-256: `c95a119978703b50488381924f3a4932eee82f8305123c90d0b01a7ade0f3c2f`
- Analysis CSV SHA-256: `94da50e64669d0e65bee4ddb99ccb26efdc7e874389982b1812a45803831aab0`

`SHA256SUMS.txt` covers the current package contents. All supplied analysis scripts, configurations, data and numerical outputs are preserved byte for byte.
