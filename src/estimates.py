"""Small estimation helpers: Kaplan–Meier medians, the naive alternatives, a back-test,
and the design matrix for the Cox model."""

from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import median_survival_times

from src import prep


def km_median(durations, observed) -> tuple[float, float, float]:
    """Kaplan–Meier median with its 95% confidence interval: (median, lower, upper).

    The interval for the median is read off the pointwise 95% band: the times where
    the band's lower and upper edges cross 50%. An edge that never reaches 50% gives
    an upper limit of infinity, meaning the data can't pin it down.
    """
    kmf = KaplanMeierFitter().fit(durations, event_observed=observed)
    lower, upper = median_survival_times(kmf.confidence_interval_).iloc[0]
    return float(kmf.median_survival_time_), float(lower), float(upper)


def naive_estimates(senators: pd.DataFrame) -> dict[str, float]:
    """The two shortcuts that ignore censoring, as means and medians.

    completed: drop everyone still serving and summarize the finished terms.
    as_ended: keep sitting senators but pretend their term ended the day the data stopped.
    """
    completed = senators.loc[senators["observed"], "years"]
    as_ended = senators["years"]
    return {
        "completed_mean": float(completed.mean()),
        "completed_median": float(completed.median()),
        "as_ended_mean": float(as_ended.mean()),
        "as_ended_median": float(as_ended.median()),
    }


def km_medians_by(senators: pd.DataFrame, column: str) -> pd.DataFrame:
    """Kaplan–Meier median and 95% CI for each group in `column`."""
    rows = []
    for group, d in senators.groupby(column, observed=True):
        median, lower, upper = km_median(d["years"], d["observed"])
        rows.append(
            {
                column: group,
                "n": len(d),
                "pct_still_serving": 100 * float((~d["observed"]).mean()),
                "km_median": median,
                "km_lower": lower,
                "km_upper": upper,
            }
        )
    return pd.DataFrame(rows).set_index(column)


def compare_methods(groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per group: the naive summaries side by side with the Kaplan–Meier median."""
    rows = []
    for label, d in groups.items():
        median, lower, upper = km_median(d["years"], d["observed"])
        still_serving = int((~d["observed"]).sum())
        rows.append(
            {
                "group": label,
                "n": len(d),
                "still_serving": still_serving,
                "pct_still_serving": 100 * still_serving / len(d),
                **naive_estimates(d),
                "km_median": median,
                "km_lower": lower,
                "km_upper": upper,
            }
        )
    return pd.DataFrame(rows).set_index("group")


def backtest(senators: pd.DataFrame, cutoff: str = "1990-01-01") -> dict:
    """Pretend the data were collected on `cutoff`, then check the estimates against reality.

    The cohort is every appointment made under the retire-at-75 rule before the cutoff.
    By the end of the data almost all of them had left, so their eventual median term
    is known and can serve as the answer key.
    """
    cutoff = pd.Timestamp(cutoff)
    cohort = senators[senators["after_1965_rule"] & (senators["start"] < cutoff)]
    snapshot = prep.censor_at(cohort, cutoff)
    median, lower, upper = km_median(snapshot["years"], snapshot["observed"])
    eventual, _, _ = km_median(cohort["years"], cohort["observed"])
    return {
        "cutoff": cutoff,
        "n": len(cohort),
        "still_serving_at_cutoff": int((~snapshot["observed"]).sum()),
        "still_serving_at_end_of_data": int((~cohort["observed"]).sum()),
        **naive_estimates(snapshot),
        "km_median": median,
        "km_lower": lower,
        "km_upper": upper,
        "eventual_median": eventual,
    }


def cox_design(senators: pd.DataFrame) -> pd.DataFrame:
    """Numeric columns for the Cox model, with readable names.

    Each categorical variable gets one 0/1 column per level except a reference level,
    so every hazard ratio reads "compared with a Liberal appointed before June 1965
    from Ontario".
    """
    design = pd.DataFrame(
        {
            "years": senators["years"],
            "observed": senators["observed"],
            "appointed after 1965 rule": senators["after_1965_rule"].astype(int),
        }
    )
    for party in ["Conservative", "Other"]:
        design[f"party: {party}"] = (senators["party_family"] == party).astype(int)
    for reg in ["Atlantic", "Quebec", "West & North"]:
        design[f"region: {reg}"] = (senators["region"] == reg).astype(int)
    return design


def fit_cox(design: pd.DataFrame, **kwargs) -> CoxPHFitter:
    """Fit a Cox proportional-hazards model on a design from `cox_design`."""
    return CoxPHFitter().fit(design, duration_col="years", event_col="observed", **kwargs)


def hazard_ratio_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Hazard ratios with 95% intervals and p-values, the columns I actually read."""
    columns = ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    return cph.summary[columns].set_axis(["hazard_ratio", "lower_95", "upper_95", "p"], axis=1)


def reason_shares(senators: pd.DataFrame) -> pd.DataFrame:
    """Percent of each period's appointments by how the term ended (rows sum to 100).

    Senators still serving get their own column instead of being dropped. Otherwise the
    shares for recent periods would come only from terms that happened to end early.
    """
    order = ["Death", "Resignation", "Retirement", prep.STILL_SERVING]
    counts = pd.crosstab(senators["period"], senators["reason"]).reindex(columns=order, fill_value=0)
    return 100 * counts.div(counts.sum(axis=1), axis=0)
