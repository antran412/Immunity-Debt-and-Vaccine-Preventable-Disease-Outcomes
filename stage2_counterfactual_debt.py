"""
STAGE 2 - Counterfactual coverage and immunity debt
Paper 2: Immunity debt and vaccine-preventable disease outcomes
------------------------------------------------------------------
PURPOSE
    Reconstruct each country's counterfactual ("no-COVID") coverage
    trajectory from the Stage 1 GLMM fixed + random effects, validate
    it on a 2019 hold-out, then compute annual immunity debt as the
    population-scaled positive gap between counterfactual and observed
    coverage.

    Counterfactual for country i, year t:
        coverage_it = (b0 + group_intercept_i + rand_intercept_i)
                    + (b_time + group_slope_i + rand_slope_i) * time_t
    Only the PRE-PANDEMIC slope is projected forward (the pandemic and
    recovery segments are the disruption we are removing).

    Immunity debt (per WHO/UNICEF surviving-infant denominator):
        gap        = max(0, counterfactual - observed)          # floor at 0
        debt       = gap/100 * surviving_infants
        debt_per1k = debt / surviving_infants * 1000

    MCV2 denominator note: MCV2 is given at 15 months-4 years, so the
    at-risk cohort is the PRIOR year's births. MCV2 uses a 1-year lagged
    surviving-infant denominator; DTP1/DTP3/MCV1 use the same-year value.

INPUT  (per vaccine <tag> in {dtp1, dtp3, mcv1, mcv2}, from Stage 1)
    <tag>_fixed.xlsx     GLMM fixed-effect estimates
    <tag>_random.xlsx    per-country random intercepts + slopes
    master_data.csv      observed coverage, surviving infants, capacity group

OUTPUT
    stage2_counterfactual_debt.csv   country-year debt for all vaccines
    validation_<tag>.csv             2019 back-test predictions per vaccine
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
MASTER_FILE   = "master_data.csv"
BASE_YEAR     = 2015          # time = year - BASE_YEAR (must match Stage 1)
VALIDATION_YR = 2019          # hold-out year for back-test
FORECAST_YRS  = range(2020, 2025)
VACCINES      = ["dtp1", "dtp3", "mcv1", "mcv2"]

# ----------------------------------------------------------------------
# Fixed-effect extraction
# ----------------------------------------------------------------------
def get_fixed(fixed_df, pattern, effect_col="Effect", est_col="Estimate"):
    """First estimate whose Effect matches `pattern` (case-insensitive); 0 if absent."""
    hit = fixed_df[fixed_df[effect_col].astype(str)
                   .str.contains(pattern, case=False, na=False, regex=True)]
    return float(hit.iloc[0][est_col]) if not hit.empty else 0.0


def build_country_params(tag, master):
    """
    Combine Stage 1 fixed + random effects into a per-country
    pre-pandemic intercept and slope for one vaccine.
    """
    fixed = pd.read_excel(f"{tag}_fixed.xlsx")
    rand  = pd.read_excel(f"{tag}_random.xlsx")

    # Clean iso3 (SAS SolutionR sometimes prefixes "iso3 " to the level)
    rand["iso3"] = (rand["iso3"].astype(str)
                    .str.replace(r"^iso3\s+", "", regex=True).str.strip())

    # Fixed effects. Reference group = "low"; HIGH/MEDIUM are offsets.
    b0     = get_fixed(fixed, r"Intercep")
    b_time = get_fixed(fixed, r"^time$|(?<![_a-z])time(?![_0-9])")

    # Group offsets for intercept and pre-pandemic slope.
    # Effect labels in ParameterEstimates carry the level name.
    b_int_high   = get_fixed(fixed, r"health_cluster\s+high")
    b_int_medium = get_fixed(fixed, r"health_cluster\s+medium")
    b_slp_high   = get_fixed(fixed, r"time\*health_cluster\s+high|"
                                    r"health_cluster\*time\s+high")
    b_slp_medium = get_fixed(fixed, r"time\*health_cluster\s+medium|"
                                    r"health_cluster\*time\s+medium")

    grp_int = {"high": b_int_high, "medium": b_int_medium, "low": 0.0}
    grp_slp = {"high": b_slp_high, "medium": b_slp_medium, "low": 0.0}

    groups = master[["iso3", "health_cluster"]].drop_duplicates()
    groups["health_cluster"] = groups["health_cluster"].str.lower().str.strip()

    cp = rand.merge(groups, on="iso3", how="left")
    missing = cp[cp["health_cluster"].isna()]["iso3"].tolist()
    if missing:
        print(f"[{tag}] WARNING: no capacity group for {len(missing)} countries: {missing}")

    cp["country_intercept"] = (
        b0
        + cp["health_cluster"].map(grp_int).fillna(0.0)
        + cp.get("rand_Intercept", 0.0)
    )
    cp["country_slope"] = (
        b_time
        + cp["health_cluster"].map(grp_slp).fillna(0.0)
        + cp.get("rand_time", 0.0)
    )
    cp["vaccine"] = tag
    return cp[["iso3", "vaccine", "health_cluster",
               "country_intercept", "country_slope"]]


# ----------------------------------------------------------------------
# Counterfactual prediction
# ----------------------------------------------------------------------
def predict(intercept, slope, year, clip=True):
    pred = intercept + slope * (year - BASE_YEAR)
    return np.clip(pred, 0, 100) if clip else pred


def validate_2019(params, master, tag):
    """Back-test: predict 2019 from the pre-pandemic trend, compare to observed."""
    obs = (master.loc[master["year"] == VALIDATION_YR, ["iso3", f"{tag}_cov"]]
           .drop_duplicates("iso3")
           .rename(columns={f"{tag}_cov": "observed"}))

    v = params.merge(obs, on="iso3", how="inner")
    v["predicted"] = predict(v["country_intercept"], v["country_slope"], VALIDATION_YR)
    v["abs_error"] = (v["predicted"] - v["observed"]).abs()

    if not v.empty:
        mae  = mean_absolute_error(v["observed"], v["predicted"])
        rmse = np.sqrt(mean_squared_error(v["observed"], v["predicted"]))
        print(f"[{tag}] 2019 back-test  n={len(v)}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    v.to_csv(f"validation_{tag}.csv", index=False)
    return v


# ----------------------------------------------------------------------
# Immunity debt
# ----------------------------------------------------------------------
def compute_debt(params, master, tag):
    """Annual immunity debt for one vaccine, 2020 onward."""
    obs = master[["iso3", "year", f"{tag}_cov", "live_birth_ths"]].copy()
    df  = obs.merge(params[["iso3", "country_intercept", "country_slope"]],
                    on="iso3", how="left")

    df["counterfactual"] = predict(df["country_intercept"], df["country_slope"], df["year"])
    df["gap"] = (df["counterfactual"] - df[f"{tag}_cov"]).clip(lower=0)
    df.loc[df["year"] < 2020, "gap"] = 0.0

    # Denominator: MCV2 uses prior-year births; others use same-year births.
    if tag == "mcv2":
        df = df.sort_values(["iso3", "year"])
        df["denom_ths"] = df.groupby("iso3")["live_birth_ths"].shift(1)
        df["denom_ths"] = df["denom_ths"].fillna(df["live_birth_ths"])
    else:
        df["denom_ths"] = df["live_birth_ths"]

    surviving = df["denom_ths"] * 1000
    df[f"debt_{tag}"]       = df["gap"] / 100 * surviving
    df[f"debt_{tag}_per1k"] = np.where(surviving > 0,
                                       df[f"debt_{tag}"] / surviving * 1000, np.nan)
    return df[["iso3", "year", f"debt_{tag}", f"debt_{tag}_per1k"]]


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    master = pd.read_csv(MASTER_FILE)
    master["iso3"] = master["iso3"].astype(str).str.strip()

    debt_tables = []
    for tag in VACCINES:
        params = build_country_params(tag, master)
        validate_2019(params, master, tag)
        debt_tables.append(compute_debt(params, master, tag))

    out = debt_tables[0]
    for t in debt_tables[1:]:
        out = out.merge(t, on=["iso3", "year"], how="outer")

    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    out.to_csv("stage2_counterfactual_debt.csv", index=False)
    print(f"\nSaved stage2_counterfactual_debt.csv  "
          f"({out['iso3'].nunique()} countries, {len(out)} rows)")


if __name__ == "__main__":
    main()
