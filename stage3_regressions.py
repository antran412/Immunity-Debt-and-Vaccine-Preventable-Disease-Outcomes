"""
STAGE 3 (main) - Immunity debt -> disease burden panel regressions
Paper 2: Immunity debt and vaccine-preventable disease outcomes
------------------------------------------------------------------
PURPOSE
    Estimate the association between accumulated immunity debt and the
    log change in disease burden above each country's pre-pandemic
    baseline. OLS with year fixed effects and standard errors clustered
    by country. Baseline subtraction acts as country fixed effects, so
    no additional country FE are included (avoids double-correction).

    For every vaccine-disease pairing and debt structure (lag1, roll2-4),
    two specifications are fit:
      (a) main effect  : logchg ~ debt + C(year)
      (b) interaction  : logchg ~ debt * C(group) + C(year)   [ref = high]
    across three samples: full, excl. Group 1, excl. Group 1+2.

INPUT   stage3_analysis_dataset.csv   (from stage3_build_dataset.py)
OUTPUT  stage3_results_main.csv       tidy coefficient table
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

# (vaccine, debt_col, outcome, disease, label)
SPECS = [
    ("MCV2", "debt_mcv2_per1k_lag1",  "logchg_measles",   "measles",   "MCV2 Lag1 -> Measles"),
    ("MCV2", "debt_mcv2_per1k_roll2", "logchg_measles",   "measles",   "MCV2 Roll2 -> Measles"),
    ("MCV2", "debt_mcv2_per1k_roll3", "logchg_measles",   "measles",   "MCV2 Roll3 -> Measles"),
    ("MCV2", "debt_mcv2_per1k_roll4", "logchg_measles",   "measles",   "MCV2 Roll4 -> Measles"),
    ("MCV1", "debt_mcv1_per1k_lag1",  "logchg_measles",   "measles",   "MCV1 Lag1 -> Measles"),
    ("MCV1", "debt_mcv1_per1k_roll2", "logchg_measles",   "measles",   "MCV1 Roll2 -> Measles"),
    ("MCV1", "debt_mcv1_per1k_roll3", "logchg_measles",   "measles",   "MCV1 Roll3 -> Measles"),
    ("MCV1", "debt_mcv1_per1k_roll4", "logchg_measles",   "measles",   "MCV1 Roll4 -> Measles"),
    ("DTP3", "debt_dtp3_per1k_lag1",  "logchg_pertussis", "pertussis", "DTP3 Lag1 -> Pertussis"),
    ("DTP3", "debt_dtp3_per1k_roll2", "logchg_pertussis", "pertussis", "DTP3 Roll2 -> Pertussis"),
    ("DTP3", "debt_dtp3_per1k_roll3", "logchg_pertussis", "pertussis", "DTP3 Roll3 -> Pertussis"),
    ("DTP3", "debt_dtp3_per1k_roll4", "logchg_pertussis", "pertussis", "DTP3 Roll4 -> Pertussis"),
]


def make_samples(df, disease):
    """Base exclusions + the three nested sample definitions."""
    base = df.copy()
    if disease == "measles":
        base = base[base["zero_measles_baseline"] == 0]
    return {
        "M1 Full":       base,
        "M2 Excl G1":    base[base["excl_group1"] == 0],
        "M3 Excl G1+2":  base[base["excl_any"]    == 0],
    }


def fit_ols(data, formula, cluster_col="iso3"):
    return smf.ols(formula, data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data[cluster_col]}
    )


def run(df, year_fe=True, years=(2020, 2023), tag="primary"):
    """Fit all specs x samples; return a tidy results DataFrame."""
    d = df[df["year"].between(*years)].copy()
    d["year"]  = d["year"].astype("category")
    d["group"] = pd.Categorical(d["group"], categories=["high", "medium", "low"])

    fe = " + C(year)" if year_fe else ""
    rows = []
    for _v, debt, outcome, disease, label in SPECS:
        for sample_name, sample in make_samples(d, disease).items():
            data = sample.dropna(subset=[debt, outcome, "group"])
            if data["iso3"].nunique() < 10:
                continue

            m = fit_ols(data, f"{outcome} ~ {debt}{fe}")
            rows.append(dict(
                spec=tag, model=label, sample=sample_name, term="main",
                coef=m.params.get(debt, np.nan),
                se=m.bse.get(debt, np.nan),
                p=m.pvalues.get(debt, np.nan),
                n_obs=int(m.nobs), n_countries=data["iso3"].nunique(),
                r2=m.rsquared,
            ))

            mi = fit_ols(data, f"{outcome} ~ {debt} * C(group){fe}")
            for term in mi.params.index:
                if debt in term:
                    rows.append(dict(
                        spec=tag, model=label, sample=sample_name, term=term,
                        coef=mi.params[term], se=mi.bse[term], p=mi.pvalues[term],
                        n_obs=int(mi.nobs), n_countries=data["iso3"].nunique(),
                        r2=mi.rsquared,
                    ))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("stage3_analysis_dataset.csv")
    res = run(df, year_fe=True, years=(2020, 2023), tag="primary_yearFE")
    res.to_csv("stage3_results_main.csv", index=False)

    sig = res[(res["term"] == "main") & (res["p"] < 0.05)]
    print(f"Saved stage3_results_main.csv  ({len(res)} rows)")
    print(f"Significant main effects (p<0.05): {len(sig)}")
    if not sig.empty:
        print(sig[["model", "sample", "coef", "se", "p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
