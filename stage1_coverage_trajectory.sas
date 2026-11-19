/*==========================================================================
  STAGE 1 — Vaccination coverage trajectories (piecewise LMM)
  Paper 2: Immunity debt and vaccine-preventable disease outcomes
--------------------------------------------------------------------------
  PURPOSE
    Fit a piecewise linear mixed model (GLMM) to routine childhood
    vaccination coverage, 2015-2024, with three time segments
    (pre-pandemic, pandemic, post-pandemic) interacted with a
    health-system capacity group. Export fixed- and random-effect
    estimates so Stage 2 (Python) can build country-specific
    counterfactual coverage trajectories.

  INPUT
    master_data.csv   One row per country-year. Required columns:
                        iso3            country ISO3 code
                        year            2015-2024
                        dtp1_cov ... mcv2_cov   coverage (0-100)
                        health_cluster  capacity group; levels must
                                        include a reference named "low"
                                        (rename in the CSV if needed)

  OUTPUT (per vaccine)
    <vaccine>_fixed.xlsx    fixed-effect parameter estimates
    <vaccine>_random.xlsx   per-country random intercepts + slopes

  NOTE ON CLASS-LEVEL ORDER  (read before trusting ESTIMATE statements)
    SAS orders CLASS levels alphabetically. With levels
    {high, low, medium} and ref="low", the two non-reference design
    columns for every  health_cluster*<time>  term are, in order:
        column 1 = HIGH      column 2 = MEDIUM
    Therefore:
        health_cluster*time 1 0   ->  HIGH
        health_cluster*time 0 1   ->  MEDIUM
    The ESTIMATE statements in %fit_vaccine below encode this mapping once, correctly, so the
    High/Medium labels cannot be swapped between vaccines. 
==========================================================================*/

/*--------------------------------------------------------------------------
  0.  Import and derive piecewise time variables
--------------------------------------------------------------------------*/
%let path = /home/u64154306/Vaccine;      /* edit to your folder */

proc import datafile="&path/master_data.csv"
     out=master_raw dbms=csv replace;
     getnames=yes;
run;

data master;
    set master_raw;

    /* Continuous time: 2015 = 0 ... 2024 = 9 */
    time = year - 2015;

    /* Piecewise segments: knots at pandemic onset (t=4) and recovery (t=6) */
    time_1 = max(0, time - 4);   /* pandemic slope change  */
    time_2 = max(0, time - 6);   /* recovery slope change  */
run;

/*--------------------------------------------------------------------------
  1.  Macro: fit one vaccine, export fixed + random effects
--------------------------------------------------------------------------
    outcome  = coverage variable (e.g. dtp3_cov)
    tag      = short name used in output filenames (e.g. dtp3)
    grp      = capacity class variable (default health_cluster)
    ref      = reference level name    (default low)
--------------------------------------------------------------------------*/
%macro fit_vaccine(outcome=, tag=, grp=health_cluster, ref=low);

    proc glimmix data=master method=laplace;
        class iso3 &grp.(ref="&ref.");

        model &outcome. =
              time time_1 time_2
              &grp.
              &grp.*time
              &grp.*time_1
              &grp.*time_2
              / solution dist=normal link=identity;

        random intercept time time_1 time_2 /
               subject=iso3 type=un solution;

        /* --- LOW (reference): segment slopes need no interaction term --- */
        estimate 'Low: pre-pandemic slope'   time 1 / e;   /* /E prints the vector */
        estimate 'Low: pandemic slope'       time 1 time_1 1;
        estimate 'Low: post-pandemic slope'  time 1 time_1 1 time_2 1;

        /* --- HIGH  (design column 1 -> "1 0") --- */
        estimate 'High: pre-pandemic slope'  time 1                      &grp.*time 1 0;
        estimate 'High: pandemic slope'      time 1 time_1 1             &grp.*time 1 0 &grp.*time_1 1 0;
        estimate 'High: post-pandemic slope' time 1 time_1 1 time_2 1    &grp.*time 1 0 &grp.*time_1 1 0 &grp.*time_2 1 0;

        /* --- MEDIUM (design column 2 -> "0 1") --- */
        estimate 'Medium: pre-pandemic slope'  time 1                    &grp.*time 0 1;
        estimate 'Medium: pandemic slope'      time 1 time_1 1           &grp.*time 0 1 &grp.*time_1 0 1;
        estimate 'Medium: post-pandemic slope' time 1 time_1 1 time_2 1  &grp.*time 0 1 &grp.*time_1 0 1 &grp.*time_2 0 1;

        ods output ParameterEstimates = fix_&tag.
                   SolutionR           = rand_&tag.;
    run;

    /* Reshape random effects: one row per country, one column per effect */
    proc transpose data=rand_&tag. out=rand_wide_&tag.(drop=_name_) prefix=rand_;
        by subject;
        id effect;
        var estimate;
    run;

    data rand_wide_&tag.;
        set rand_wide_&tag.;
        rename subject = iso3;
    run;

    proc sort data=rand_wide_&tag.; by iso3; run;

    proc export data=fix_&tag.
        outfile="&path/&tag._fixed.xlsx"  dbms=xlsx replace;
    run;
    proc export data=rand_wide_&tag.
        outfile="&path/&tag._random.xlsx" dbms=xlsx replace;
    run;

%mend fit_vaccine;

/*--------------------------------------------------------------------------
  2.  Run the four primary models (capacity cluster as predictor)
--------------------------------------------------------------------------*/
%fit_vaccine(outcome=dtp1_cov, tag=dtp1);
%fit_vaccine(outcome=dtp3_cov, tag=dtp3);
%fit_vaccine(outcome=mcv1_cov, tag=mcv1);
%fit_vaccine(outcome=mcv2_cov, tag=mcv2);

/*--------------------------------------------------------------------------
  3.  OPTIONAL — alternative capacity definitions (Supplementary)
      Re-run any vaccine with a different grouping variable, e.g.:
        %fit_vaccine(outcome=mcv2_cov, tag=mcv2_uhc,    grp=uhc_group);
        %fit_vaccine(outcome=mcv2_cov, tag=mcv2_doctor, grp=doctor_group);
      Compare model fit (AIC/BIC) across grouping choices; the composite
      cluster was selected for the main analysis.
--------------------------------------------------------------------------*/
