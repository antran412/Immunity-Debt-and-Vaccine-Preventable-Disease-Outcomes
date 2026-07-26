"""
STAGE 3 (build) - Assemble the panel analysis dataset
Paper 2: Immunity debt and vaccine-preventable disease outcomes
------------------------------------------------------------------
PURPOSE
    Combine Stage 2 immunity-debt estimates with disease-burden data to
    build the country-year panel used in the Stage 3 regressions.
    Produces, per country-year:
      - annual, lag-1, and rolling-sum (2/3/4-yr) debt for DTP3/MCV1/MCV2
      - pre-pandemic baseline (2015-2019 mean) for measles and pertussis
      - log-change outcome = log(rate+1) - log(baseline+1)
      - zero-baseline measles flag and sensitivity exclusion flags

INPUT
    stage2_counterfactual_debt.csv   annual debt per vaccine (Stage 2)
    master_data.csv                  disease rates, region, capacity group,
                                     country name
                                     (ihme_mearate, ihme_perrate,
                                      measles_rate, per_rate for WHO)

OUTPUT
    stage3_analysis_dataset.csv      panel dataset for the regressions
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# Sensitivity exclusion groups (see Supplementary methods)
GROUP1 = ["MLI", "COG", "NGA", "LBR", "PNG", "ETH", "VEN", "MEX",
          "ATG", "DOM", "BRA", "FJI", "TGO", "ARG", "DMA"]   # validation-error
GROUP2 = ["COD", "GIN", "CMR", "TCD"]                        # pre-existing burden


def add_lag_and_rolling(df, vaccines=("dtp3", "mcv1", "mcv2")):
    """Lag-1 and rolling-sum (2/3/4-year) debt structures per country."""
    for v in vaccines:
        col = f"debt_{v}_per1k"
        df[f"{col}_lag1"] = df.groupby("iso3")[col].shift(1)
        for w in (2, 3, 4):
            df[f"{col}_roll{w}"] = (
                df.groupby("iso3")[col]
                  .rolling(w, min_periods=w).sum()
                  .reset_index(level=0, drop=True)
            )
    return df


def add_outcomes(df):
    """Pre-pandemic (2015-2019) baseline and log-change outcomes (IHME)."""
    baseline = (
        df[df["year"].between(2015, 2019)]
        .groupby("iso3")[["ihme_mearate", "ihme_perrate"]].mean()
        .rename(columns={"ihme_mearate": "baseline_mearate",
                         "ihme_perrate": "baseline_perrate"})
        .reset_index()
    )
    df = df.merge(baseline, on="iso3", how="left")

    df["logchg_measles"]   = np.log1p(df["ihme_mearate"]) - np.log1p(df["baseline_mearate"])
    df["logchg_pertussis"] = np.log1p(df["ihme_perrate"]) - np.log1p(df["baseline_perrate"])

    zero_measles = baseline.loc[baseline["baseline_mearate"] == 0, "iso3"].tolist()
    df["zero_measles_baseline"] = df["iso3"].isin(zero_measles).astype(int)
    print(f"Zero measles-baseline countries excluded from measles models: {len(zero_measles)}")
    return df


def add_exclusion_flags(df):
    df["excl_group1"] = df["iso3"].isin(GROUP1).astype(int)
    df["excl_group2"] = df["iso3"].isin(GROUP2).astype(int)
    df["excl_any"]    = df["iso3"].isin(set(GROUP1 + GROUP2)).astype(int)
    return df


def main():
    debt   = pd.read_csv("stage2_counterfactual_debt.csv")
    master = pd.read_csv("master_data.csv")

    df = master.merge(debt, on=["iso3", "year"], how="left")
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    for c in ("region", "group"):
        if c in df:
            df[c] = df.groupby("iso3")[c].ffill().bfill()

    df = add_lag_and_rolling(df)
    df = add_outcomes(df)
    df = add_exclusion_flags(df)

    keep = [
        "iso3", "country", "year", "region", "group",
        "ihme_mearate", "ihme_perrate", "measles_rate", "per_rate",
        "baseline_mearate", "baseline_perrate",
        "logchg_measles", "logchg_pertussis",
        *[f"debt_{v}_per1k{suf}"
          for v in ("dtp3", "mcv1", "mcv2")
          for suf in ("", "_lag1", "_roll2", "_roll3", "_roll4")],
        "zero_measles_baseline", "excl_group1", "excl_group2", "excl_any",
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_csv("stage3_analysis_dataset.csv", index=False)
    print(f"Saved stage3_analysis_dataset.csv  ({len(df)} rows, {len(keep)} cols)")


if __name__ == "__main__":
    main()
