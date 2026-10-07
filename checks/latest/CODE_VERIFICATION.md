# Latest code-verification report

The audit examined the final research supplement in an isolated copy. All checked scenarios passed; the original seven Python scripts were preserved without modification.

## Executed checks

| Check | Outcome | Evidence |
|---|---|---|
| Source archive integrity | ZIP CRC and all 178 source-manifest entries passed | Supplied archive examined before extraction |
| Existing unit tests | 8/8 passed | `unit_tests.txt` |
| Saved predictions and attribution arrays | Split boundaries, ensemble means, Shapley additivity and full-retention equality passed | `saved_verification.txt` |
| Main real-data split audit | 117 outer/internal boundaries over 30 outer folds passed | `extended_verification.txt` |
| Processed CSV reconstruction | Exact match for 1,360 records and 683 derived units, including missing entries | Reconstructed with the supplied `work/build_analysis.py` |
| Metrics, risk–retention curves, family effects and bootstrap interval | Reconstructed from saved results | `work/revision/verification/`, relative to the package root |
| Predictive preprocessing controls | All 5,610 predictions, 16 metric summaries, split statistics and family audits reproduced exactly | `ablation_rerun.txt`, `ablation_exact_comparison.txt` |
| Diagnostic-score reconstruction | All 30 outer-fold g0/g1 scores restored to machine precision; maximum absolute difference below 8 × 10⁻¹⁵ | `diagnostic_reconstruction.csv` |
| Synthetic full nested run | Four protocols completed, 18 outer folds, 300 predictions and 54 outer/internal boundaries | `smoke_all_protocols.txt`, `smoke_verification.txt`, `smoke/` |
| Numerical publication figures | All three PNG charts pixel-identical to the provided figures | `figure_reproduction.txt` records the successful plotting invocation; comparison executed during the audit |

An empty `figure_reproduction.txt` is the plotting script's successful standard-output log; the script produces image files.

## Method implementation reviewed

Imputation and scaling are fitted on the applicable training data. The diagnostic g0/g1 models use inner out-of-fold errors and are frozen before external evaluation. Background records and bootstrap groups belong to the relevant training set. The permutation-based attribution estimator was checked on analytic linear and pairwise-interaction examples and for additive reconstruction of saved predictions. Retention arithmetic and score-tie resolution are independent of target values and input row order.

## Scope of recomputation

The real-data preprocessing controls were retrained. The main real-data metrics and diagnostic models were reconstructed from the saved outer/internal OOF outputs and attribution arrays. The complete nested training-and-attribution workflow was additionally run on the synthetic test fixture under all four protocols. The original main SteelBench experiment with all attribution calculations was not fully refitted during this audit.

The synthetic fixture and its outputs are software-validation evidence. The published SteelBench results reside in `work/real_run/results_uts_fast/` and `work/revision/`.

## Confirmed primary result

Mean family-level `AURC(g0) − AURC(g1)` = **4.277404158263354 MPa**; conditional 95% family-bootstrap interval approximately **[−0.47867820347143, 9.84477954244799] MPa**. These reproduce the rounded manuscript values.

## Environment

Python 3.12.14; NumPy 2.3.5; pandas 2.2.3; SciPy 1.17.0; scikit-learn 1.8.0; matplotlib 3.10.8; joblib 1.5.3. The six numerical-package versions match the pinned requirements.

See the root `README.md` for verification and reproduction commands. All numerical software-test outcomes are interpreted within the scope described above.
