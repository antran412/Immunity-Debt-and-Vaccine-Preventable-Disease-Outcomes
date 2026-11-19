# Immunity Debt and Vaccine-Preventable Disease Outcomes — Analysis Code

Analysis code for the three-stage study linking pandemic-era routine
childhood vaccination disruption to immunity debt and vaccine-preventable
disease burden across 179 countries (2015–2024).

The pipeline runs in three stages. Stage 1 and the counterfactual step use
**SAS**; the remaining computation is **Python**.

```
Stage 1 (SAS)        Stage 2 (Python)            Stage 3 (Python)
coverage LMM   -->  counterfactual + debt  -->  panel regressions
fixed+random         validation + immunity        debt -> disease burden
effects export       debt per country-year        + sensitivity checks
```

## Repository layout

```
.
├── stage1_sas/
│   └── stage1_coverage_trajectory.sas   Piecewise GLMM; exports fixed + random effects
├── stage2_python/
│   └── stage2_counterfactual_debt.py    Counterfactual coverage, 2019 back-test, immunity debt
├── stage3_python/
│   ├── stage3_build_dataset.py          Assemble panel (debt + baseline + outcomes + flags)
│   ├── stage3_regressions.py            Primary models (year FE, clustered SE)
│   └── stage3_sensitivity.py            Option 1, Option 2, and WHO robustness checks
├── requirements.txt
└── README.md
```

## Data

Analyses use publicly available data; raw sources are not redistributed here.

| Domain | Source |
| --- | --- |
| Routine childhood vaccination coverage (DTP1, DTP3, MCV1, MCV2) | WHO/UNICEF WUENIC |
| Health-system capacity (UHC index, physician & nurse density, health expenditure %GDP) | WHO GHO |
| Disease burden (measles, pertussis) | IHME GBD; WHO surveillance |
| Surviving infants, population | UN World Population Prospects |
| Income group, region | World Bank |

The scripts expect a single prepared panel, `master_data.csv`, with one row
per country-year and at least these columns:

```
iso3, country, year, region, group (or health_cluster),
dtp1_cov, dtp3_cov, mcv1_cov, mcv2_cov,
live_birth_ths,
ihme_mearate, ihme_perrate, measles_rate, per_rate
```

`group` / `health_cluster` is the composite health-system capacity cluster
with levels `low`, `medium`, `high` (`low` is the reference).

## How to run

### Stage 1 — coverage trajectories (SAS)

Edit the `%let path=` line, place `master_data.csv` there, and run
`stage1_coverage_trajectory.sas`. For each vaccine it writes
`<tag>_fixed.xlsx` and `<tag>_random.xlsx`.

> **Class-level order.** SAS orders CLASS levels alphabetically. With levels
> `{high, low, medium}` and `ref="low"`, the two interaction design columns
> are, in order, **high** then **medium** — so `... *time 1 0` is HIGH and
> `... *time 0 1` is MEDIUM. The macro encodes this once. The first ESTIMATE
> carries the `/E` option: check its printed coefficient vector in the log to
> confirm the column order on your data before trusting the rest.

### Stage 2 — counterfactual and immunity debt (Python)

```bash
cd stage2_python
python stage2_counterfactual_debt.py
```

Reads the Stage 1 `*_fixed.xlsx` / `*_random.xlsx` and `master_data.csv`;
writes `stage2_counterfactual_debt.csv` and `validation_<tag>.csv`
(2019 back-test with MAE/RMSE per vaccine).

### Stage 3 — panel regressions (Python)

```bash
cd stage3_python
python stage3_build_dataset.py     # -> stage3_analysis_dataset.csv
python stage3_regressions.py       # -> stage3_results_main.csv
python stage3_sensitivity.py       # -> option1 / option2 / who result tables
```

Keep the three Stage 3 scripts in the same folder: `stage3_sensitivity.py`
imports the shared model runner from `stage3_regressions.py`.

## Method notes

- **Piecewise time.** `time = year − 2015`; knots at pandemic onset (t=4) and
  recovery (t=6) via `time_1 = max(0, time−4)`, `time_2 = max(0, time−6)`.
- **Immunity debt.** `debt = max(0, counterfactual − observed)/100 ×
  surviving_infants`; floored at zero (debt accumulates, never negative);
  reported per 1,000 surviving infants.
- **MCV2 denominator.** MCV2 is given at 15 months–4 years, so it uses the
  **prior-year** surviving-infant cohort; DTP1/DTP3/MCV1 use the same-year cohort.
- **Log-change outcome.** `log(rate+1) − log(baseline+1)`, where baseline is
  the 2015–2019 country mean. Baseline subtraction acts as country fixed
  effects, so country FE are not added again.
- **Year FE vs baseline subtraction.** In the primary model year FE absorb the
  global NPI suppression–rebound trend, leaving little variance for debt
  (Option 2 removes year FE to avoid this double-correction).
- **Sensitivity samples.** Group 1 = validation-error countries; Group 2 =
  pre-existing-burden countries; models are refit excluding each.

## Software

- SAS 9.4 (`PROC GLIMMIX`, `method=laplace`)
- Python ≥ 3.9 — see `requirements.txt`

## Not included

Data-acquisition scripts are omitted because inputs come from many public
portals with their own access terms. See the table above for sources; the
prepared `master_data.csv` schema is documented so the pipeline is reproducible
once inputs are assembled.
