# Preprocessing controls and supporting results

Run the commands from the package root; the full software/environment instructions are in the root `README.md`.

| File/directory | Contents |
|---|---|
| `ablation_predictions.csv` | 5,610 predictions from median imputation, missingness indicators, omission of the four most incomplete predictors, and complete-case controls |
| `ablation_summary.csv` | Sixteen protocol/strategy summaries |
| `split_statistics.csv` | Train/test counts for all 30 outer folds, with record, family, grade and split-unit counts |
| `family_audit.csv`, `family_source.csv`, `audit.json` | Family size, source composition and preprocessing summaries |
| `verification/` | Checked boundaries, exact predictor overlap, reconstructed metrics and primary effect |
| `figures/` | Workflow illustration and numerical figures used in the manuscript |
| `literature_identifiers.md` | Literature identifiers matching the manuscript reference order |

## Reproduction

```sh
python work/revision_analysis.py
python work/revision_checks.py
python work/plot_revision_figures.py
```

The preprocessing controls use the fixed outer-fold assignments from the main experiment. They assess predictive UTS error under alternative preprocessing strategies. The attribution-augmented diagnostic comparison is based on the original median-imputed ensemble and its saved internal OOF results.

Complete-case selection retains 169 records from two families and one source tier. The grade-held-out control compares median and complete-case training on the same 169 eligible test records. Its evaluation population differs from the 1,360-record population used for the main results.

The main family-level diagnostic effect and resampling interval are in `../real_run/results_uts_fast/primary_result.json`. The latest verification evidence is in `../../checks/latest/`.
