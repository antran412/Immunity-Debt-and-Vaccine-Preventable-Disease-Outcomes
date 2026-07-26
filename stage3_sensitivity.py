"""
STAGE 3 (sensitivity) - Robustness specifications
Paper 2: Immunity debt and vaccine-preventable disease outcomes
------------------------------------------------------------------
PURPOSE
    Three robustness checks on the Stage 3 debt -> disease models:

    Option 1  Post-NPI period only (2022-2023). Drops 2020-2021, when
              disease was globally suppressed by lockdowns, isolating the
              rebound window where the debt-disease signal can emerge.

    Option 2  No year fixed effects (2020-2023). Baseline subtraction
              already removes country-level pre-pandemic differences;
              adding year FE double-corrects and absorbs the debt signal
              (DTP3 R^2 falls from ~0.58 to ~0.01 when year FE are added).

    WHO check Re-run using WHO surveillance rates (measles_rate, per_rate)
              instead of IHME. Expected to be directionally consistent but
              noisier, quantifying surveillance bias.

    Options 1 and 2 reuse the model runner in stage3_regressions.py.

INPUT   stage3_analysis_dataset.csv   (from stage3_build_dataset.py)
OUTPUT  stage3_results_option1.csv
        stage3_results_option2.csv
        stage3_results_who.csv
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
warnings.filterwarnings("ignore")

from stage3_regressions import run, make_samples, fit_ols

# WHO version of the specs (surveillance-rate outcomes)
WHO_SPECS = [
    ("MCV2", "debt_mcv2_per1k_lag1",  "logchg_who_measles",   "measles",   "MCV2 Lag1 -> Measles (WHO)"),
    ("MCV2", "debt_mcv2_per1k_roll3", "logchg_who_measles",   "measles",   "MCV2 Roll3 -> Measles (WHO)"),
    ("MCV2", "debt_mcv2_per1k_roll4", "logchg_who_measles",   "measles",   "MCV2 Roll4 -> Measles (WHO)"),
    ("DTP3", "debt_dtp3_per1k_roll3", "logchg_who_pertussis", "pertussis", "DTP3 Roll3 -> Pertussis (WHO)"),
    ("DTP3", "debt_dtp3_per1k_roll4", "logchg_who_pertussis", "pertussis", "DTP3 Roll4 -> Pertussis (WHO)"),
]


def build_who_outcomes(df):
    """Add WHO log-change outcomes from surveillance rates."""
    base = (
        df[df["year"].between(2015, 2019)]
        .groupby("iso3")[["measles_rate", "per_rate"]].mean()
        .rename(columns={"measles_rate": "base_who_mea", "per_rate": "base_who_per"})
        .reset_index()
    )
    df = df.merge(base, on="iso3", how="left")
    df["logchg_who_measles"]   = np.log1p(df["measles_rate"]) - np.log1p(df["base_who_mea"])
    df["logchg_who_pertussis"] = np.log1p(df["per_rate"])     - np.log1p(df["base_who_per"])
    zero = base.loc[base["base_who_mea"] == 0, "iso3"].tolist()
    df["zero_measles_baseline"] = df["iso3"].isin(zero).astype(int)
    return df


def run_who(df, year_fe=False, years=(2020, 2023)):
    d = df[df["year"].between(*years)].copy()
    d["year"]  = d["year"].astype("category")
    d["group"] = pd.Categorical(d["group"], categories=["high", "medium", "low"])
    fe = " + C(year)" if year_fe else ""

    rows = []
    for _v, debt, outcome, disease, label in WHO_SPECS:
        for sample_name, sample in make_samples(d, disease).items():
            data = sample.dropna(subset=[debt, outcome, "group"])
            if data["iso3"].nunique() < 10:
                continue
            m = fit_ols(data, f"{outcome} ~ {debt}{fe}")
            rows.append(dict(
                model=label, sample=sample_name,
                coef=m.params.get(debt, np.nan), se=m.bse.get(debt, np.nan),
                p=m.pvalues.get(debt, np.nan),
                n_obs=int(m.nobs), n_countries=data["iso3"].nunique(),
            ))
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv("stage3_analysis_dataset.csv")

    # Option 1: post-NPI window, keep year FE
    opt1 = run(df, year_fe=True, years=(2022, 2023), tag="option1_postNPI")
    opt1.to_csv("stage3_results_option1.csv", index=False)

    # Option 2: full pandemic window, drop year FE
    opt2 = run(df, year_fe=False, years=(2020, 2023), tag="option2_noYearFE")
    opt2.to_csv("stage3_results_option2.csv", index=False)

    # WHO surveillance check (three specifications)
    who_df = build_who_outcomes(df)
    who = pd.concat([
        run_who(who_df, year_fe=True,  years=(2020, 2023)).assign(spec="who_primary"),
        run_who(who_df, year_fe=True,  years=(2022, 2023)).assign(spec="who_option1"),
        run_who(who_df, year_fe=False, years=(2020, 2023)).assign(spec="who_option2"),
    ], ignore_index=True)
    who.to_csv("stage3_results_who.csv", index=False)

    print("Saved: stage3_results_option1.csv, stage3_results_option2.csv, stage3_results_who.csv")
    for name, r in [("Option 1", opt1), ("Option 2", opt2)]:
        sig = r[(r["term"] == "main") & (r["p"] < 0.05)]
        print(f"{name}: {len(sig)} significant main effects (p<0.05)")


if __name__ == "__main__":
    main()
