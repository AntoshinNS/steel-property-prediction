# SteelBench v1.0 — Dataset Card

## Overview

| Field | Value |
|-------|-------|
| **Name** | SteelBench v1.0 |
| **Version** | 1.0 (2026-02-02) |
| **File** | `steelbench_core.csv` |
| **Shape** | 1,636 rows x 25 columns |
| **Grades** | 594 unique steel grades |
| **Families** | 17 steel families |
| **Primary target** | `tensile_strength` (UTS, MPa) — 100% complete |
| **Secondary target** | `yield_strength` (YS, MPa) — 71.6% complete |

## Schema

| Column | Type | Description | Missingness |
|--------|------|-------------|-------------|
| `heat_id` | str | Unique sample identifier | 0% |
| `grade_id` | str | Steel grade (594 unique) | 0% |
| `source` | str | Original data source name | 0% |
| `C` | float | Carbon (wt%) | 0.2% |
| `Mn` | float | Manganese (wt%) | 1.0% |
| `Si` | float | Silicon (wt%) | 12.7% |
| `Cr` | float | Chromium (wt%) | 11.3% |
| `Ni` | float | Nickel (wt%) | 18.8% |
| `Mo` | float | Molybdenum (wt%) | 26.3% |
| `V` | float | Vanadium (wt%) | 0% |
| `Cu` | float | Copper (wt%) | 36.4% |
| `Al` | float | Aluminum (wt%) | 49.4% |
| `austenitize_T` | float | Austenitization temp (C) | 41.1% |
| `temper_T` | float | Tempering temp (C) | 60.6% |
| `quench_medium` | str | Quench medium (water/oil/air) | 44.1% |
| `tensile_strength` | float | UTS (MPa) — primary target | 0% |
| `yield_strength` | float | YS (MPa) | 28.4% |
| `elongation` | float | Elongation (%) | 46.7% |
| `reduction_area` | float | Reduction of area (%) | 50.6% |
| `impact_J_avg` | float | Charpy impact energy (J) | 55.3% |
| `condition` | str | Heat treatment condition | 86.8% |
| `hardness` | float | Hardness value | 27.2% |
| `split` | str | Predefined train/val/test split | 0% |
| `data_tier` | str | Provenance tier | 0% |
| `steel_family` | str | Steel family (17 unique) | 16.9% |

## License

**CC-BY 4.0** (Creative Commons Attribution 4.0 International)

See [LICENSE](LICENSE) for full terms.

## Provenance / Data Tiers

| Tier | Rows | Description | Original License |
|------|------|-------------|------------------|
| `emk_spec_verified` | 753 | EMK handbook — specification minimums | Public domain |
| `nims_measured` | 360 | NIMS (Japan) — measured lab data | Public domain (gov) |
| `kaggle_measured` | 247 | Kaggle datasets — measured values | CC-BY / PD |

**Note:** This open release excludes restricted tiers (MMPDS, internal lab data).

## Known Issues

1. **EMK rows are spec minimums, not measured values.** The `emk_spec_verified` tier (46% of data) contains specification minimum values, not actual measurements. This creates distribution shift vs. measured data.
2. **MMPDS license uncertainty.** MMPDS handbook data may be restricted. The `steelbench_v1.0_open` release excludes MMPDS rows for license safety.
3. **High missingness in process features.** `temper_T` (61%), `Al` (49%), `quench_medium` (44%) are frequently missing. Models must handle missing values.
4. **Steel family coverage.** `steel_family` is missing for 17% of rows. These rows are excluded from LOFO evaluation.
5. **No microstructure features.** The dataset contains only composition + heat treatment. Microstructure (grain size, phase fractions) is not included.

## Evaluation Protocols

| Protocol | Group Column | k | Use Case |
|----------|-------------|---|----------|
| RandomKFold | — | 5 | Standard baseline |
| GKF-grade | `grade_id` | 5 | Primary grade-shift metric |
| LOFO-family | `steel_family` | 12 | Family-level generalization |
| LOSO-source | `data_tier` | 5 | Cross-source generalization |
| LOGO-grade | `grade_id` | all | Most stringent (leave-one-grade-out) |

## Headline Metric

**OOF-global R^2**: Single R^2 computed from all concatenated out-of-fold predictions. This avoids the pitfalls of mean-of-fold R^2 when fold sizes are imbalanced.

## Ethics & Intended Use

- This dataset is intended for **materials informatics research** — predicting steel mechanical properties from composition and heat treatment.
- The dataset should **not** be used for safety-critical decisions without independent validation. Model predictions are not a substitute for physical testing.
- No personally identifiable information is included.

## Citation

```
@inproceedings{steelbench2026,
  title={SteelBench: Predicting Steel Properties under Real-World Data Gaps, Specification Ranges, and Grade Shift},
  author={Anonymous},
  booktitle={KDD},
  year={2026}
}
```
