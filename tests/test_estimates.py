import math

import pandas as pd
import pytest

from src import estimates, prep


def _toy(durations, observed):
    return pd.DataFrame({"years": durations, "observed": observed})


def test_km_median_equals_plain_median_when_nothing_is_censored():
    median, _, _ = estimates.km_median([1, 2, 3, 4, 5], [True] * 5)
    assert median == 3


def test_km_median_by_hand_with_censoring():
    # Six terms; two senators still serving at 3 years.
    #   t=2: 1 of 6 leaves      -> S = 5/6
    #   t=3: 2 censored, they leave the risk set without counting as exits
    #   t=4: 1 of 3 leaves      -> S = 5/6 * 2/3 = 0.556
    #   t=6: 1 of 2 leaves      -> S = 0.556 * 1/2 = 0.278  (first time S <= 0.5)
    durations = [2, 4, 6, 8, 3, 3]
    observed = [True, True, True, True, False, False]
    median, lower, upper = estimates.km_median(durations, observed)
    assert median == 6
    assert lower <= median
    assert math.isinf(upper) or upper >= median


def test_naive_estimates_understate_on_the_same_toy_data():
    toy = _toy([2, 4, 6, 8, 3, 3], [True, True, True, True, False, False])
    naive = estimates.naive_estimates(toy)
    assert naive["completed_median"] == 5.0    # median of 2, 4, 6, 8
    assert naive["as_ended_median"] == 3.5     # median of 2, 3, 3, 4, 6, 8
    assert naive["completed_mean"] == 5.0
    assert naive["as_ended_mean"] == pytest.approx(26 / 6)
    km, _, _ = estimates.km_median(toy["years"], toy["observed"])
    assert km > naive["completed_median"] > naive["as_ended_median"]


def test_compare_methods_reports_censoring_share():
    toy = _toy([2, 4, 6, 8, 3, 3], [True, True, True, True, False, False])
    table = estimates.compare_methods({"toy": toy})
    row = table.loc["toy"]
    assert row["n"] == 6
    assert row["still_serving"] == 2
    assert row["pct_still_serving"] == pytest.approx(100 * 2 / 6)
    assert row["km_median"] == 6


def test_km_medians_by_group():
    toy = pd.DataFrame(
        {
            "years": [1, 2, 3, 2, 4, 6, 8, 3, 3],
            "observed": [True] * 3 + [True, True, True, True, False, False],
            "group": ["a"] * 3 + ["b"] * 6,
        }
    )
    table = estimates.km_medians_by(toy, "group")
    assert table.loc["a", "km_median"] == 2
    assert table.loc["b", "km_median"] == 6
    assert table.loc["b", "pct_still_serving"] == pytest.approx(100 * 2 / 6)


def test_reason_shares_rows_sum_to_100_and_keep_sitting_senators():
    toy = pd.DataFrame(
        {
            "period": pd.Categorical(["p1", "p1", "p2", "p2"]),
            "reason": ["Death", "Resignation", "Retirement", prep.STILL_SERVING],
        }
    )
    shares = estimates.reason_shares(toy)
    assert shares.sum(axis=1).tolist() == [100.0, 100.0]
    assert shares.loc["p2", prep.STILL_SERVING] == 50.0
    assert shares.loc["p1", "Retirement"] == 0.0


def test_cox_design_uses_liberal_ontario_before_1965_as_reference():
    senators = pd.DataFrame(
        {
            "years": [10.0, 12.0, 3.0],
            "observed": [True, True, False],
            "after_1965_rule": [False, True, True],
            "party_family": ["Liberal", "Conservative", "Other"],
            "region": ["Ontario", "Quebec", "West & North"],
        }
    )
    design = estimates.cox_design(senators)
    dummies = design.drop(columns=["years", "observed"])
    assert dummies.iloc[0].sum() == 0  # the reference senator has every dummy at 0
    assert dummies.iloc[1].to_dict() == {
        "appointed after 1965 rule": 1,
        "party: Conservative": 1,
        "party: Other": 0,
        "region: Atlantic": 0,
        "region: Quebec": 1,
        "region: West & North": 0,
    }


def test_backtest_on_toy_data():
    starts = pd.to_datetime(["1966-01-01", "1970-01-01", "1980-01-01", "1995-01-01"])
    ends = pd.to_datetime(["1976-01-01", "1990-01-01", "1999-01-01", "2010-01-01"])
    senators = pd.DataFrame(
        {
            "start": starts,
            "end": ends,
            "observed": [True, True, True, True],
            "reason": ["Death"] * 4,
            "years": prep.days_to_years((ends - starts).days),
            "after_1965_rule": [True] * 4,
        }
    )
    result = estimates.backtest(senators, cutoff="1985-01-01")
    assert result["n"] == 3                       # the 1995 appointment is outside the cohort
    assert result["still_serving_at_cutoff"] == 2  # the 1970 and 1980 terms were running in 1985
    assert result["still_serving_at_end_of_data"] == 0
