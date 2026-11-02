"""Every number quoted in the README, recomputed from the data.

If a new lifelines release changes the bundled data or the estimators, these tests fail
and tell me the README needs updating.
"""

import pandas as pd
import pytest
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test, proportional_hazard_test

from src import estimates, prep


def one_decimal(x):
    return round(float(x), 1)


@pytest.fixture(scope="module")
def raw():
    return prep.load_raw()


@pytest.fixture(scope="module")
def senators(raw):
    return prep.prepare(raw)


@pytest.fixture(scope="module")
def before(senators):
    return senators[~senators["after_1965_rule"]]


@pytest.fixture(scope="module")
def after(senators):
    return senators[senators["after_1965_rule"]]


def test_data_section(raw, senators):
    assert (len(raw), raw["Name"].nunique(), len(senators)) == (933, 922, 930)
    assert int((~senators["observed"]).sum()) == 99
    assert prep.censoring_date(raw) == pd.Timestamp("2013-10-01")
    territories = raw["Province / Territory"].isin(["Yukon", "Northwest Territories", "Nunavut"])
    assert territories.sum() == 11


def test_finding_1_overall_median(senators):
    median, lower, upper = estimates.km_median(senators["years"], senators["observed"])
    assert (one_decimal(median), one_decimal(lower), one_decimal(upper)) == (14.0, 13.0, 14.8)


def pick(row, keys):
    """Several values from a row or dict, rounded to one decimal like the README."""
    return [one_decimal(row[k]) for k in keys]


MEANS = ["completed_mean", "as_ended_mean"]
MEDIANS = ["completed_median", "as_ended_median"]
KM = ["km_median", "km_lower", "km_upper"]


def test_finding_2_naive_vs_km(senators, before, after):
    table = estimates.compare_methods({"all": senators, "before": before, "after": after})
    assert pick(table.loc["all"], MEANS + MEDIANS + KM) == [14.5, 13.8, 13.3, 12.5, 14.0, 13.0, 14.8]
    assert pick(table.loc["before"], MEANS + MEDIANS + KM) == [15.2, 15.2, 13.7, 13.7, 13.7, 12.9, 14.9]
    assert pick(table.loc["after"], MEANS + MEDIANS + KM) == [12.6, 11.4, 11.4, 9.9, 14.2, 12.5, 15.0]
    assert pick(table["pct_still_serving"], ["all", "before", "after"]) == [10.6, 0.0, 29.7]
    assert one_decimal(senators["years"].max()) == 48.5


def test_finding_3_backtest_1990(senators):
    back = estimates.backtest(senators, cutoff="1990-01-01")
    counts = [back["n"], back["still_serving_at_cutoff"], back["still_serving_at_end_of_data"]]
    assert counts == [142, 86, 3]
    assert pick(back, MEDIANS) == [8.9, 10.2]
    assert pick(back, KM) == [15.8, 13.7, 19.1]
    assert one_decimal(back["eventual_median"]) == 15.0
    # The point of the back-test: the 1990 interval contains what actually happened.
    assert back["km_lower"] <= back["eventual_median"] <= back["km_upper"]
    assert one_decimal(back["km_median"] - back["eventual_median"]) == 0.8
    assert pick({"a": 15.0 - back["completed_median"], "b": 15.0 - back["as_ended_median"]}, "ab") == [6.1, 4.8]
    # The answer key is exact: the 3 senators still serving in 2013 had all served more than
    # 29 years, far past the median, so their unknown end dates can't move it.
    cohort = senators[senators["after_1965_rule"] & (senators["start"] < "1990-01-01")]
    assert (cohort.loc[~cohort["observed"], "years"] > 29).all()


def test_finding_4_eras(senators, before, after):
    assert one_decimal(estimates.km_median(before["years"], before["observed"])[0]) == 13.7
    test = logrank_test(before["years"], after["years"], before["observed"], after["observed"])
    assert round(test.p_value, 2) == 0.28

    still = {}
    for name, d in [("before", before), ("after", after)]:
        kmf = KaplanMeierFitter().fit(d["years"], d["observed"])
        still[name] = [round(100 * s, 1) for s in kmf.survival_function_at_times([20, 30, 35]).values]
    assert still["before"] == [30.2, 8.5, 4.5]
    assert still["after"] == [25.5, 8.1, 0.7]
    assert (one_decimal(before["years"].max()), one_decimal(after["years"].max())) == (48.5, 36.9)

    periods = multivariate_logrank_test(senators["years"], senators["period"], senators["observed"])
    assert round(periods.p_value, 2) == 0.14
    by_period = estimates.km_medians_by(senators, "period")
    assert by_period["km_median"].map(one_decimal).tolist() == [12.5, 13.2, 14.6, 14.2, 15.0, 11.6]
    assert by_period.loc["1990–2013", "pct_still_serving"].round() == 50


def two_decimals(row):
    return [round(float(row[c]), 2) for c in ["hazard_ratio", "lower_95", "upper_95"]]


def test_finding_5_cox_model(senators):
    design = estimates.cox_design(senators)
    cph = estimates.fit_cox(design)
    table = estimates.hazard_ratio_table(cph)
    assert (table["lower_95"] < 1).all() and (table["upper_95"] > 1).all()  # every CI crosses 1
    assert round(cph.concordance_index_, 2) == 0.53
    assert round(cph.log_likelihood_ratio_test().p_value, 2) == 0.53
    assert two_decimals(table.loc["appointed after 1965 rule"]) == [1.11, 0.95, 1.30]
    assert two_decimals(table.loc["party: Conservative"]) == [1.02, 0.88, 1.17]
    assert (senators["party_family"] == "Other").sum() == 17

    # Proportional-hazards check: exactly two terms flagged, both at p = 0.02.
    ph = proportional_hazard_test(cph, design, time_transform="rank").summary["p"]
    assert sorted(ph[ph < 0.05].index) == ["party: Other", "region: West & North"]
    assert ph[ph < 0.05].round(2).tolist() == [0.02, 0.02]

    # Stratified by region: estimates barely move and nothing is flagged.
    by_region = design.drop(columns=[c for c in design.columns if c.startswith("region:")])
    by_region["region"] = senators["region"]
    stratified = estimates.fit_cox(by_region, strata=["region"])
    era = estimates.hazard_ratio_table(stratified).loc["appointed after 1965 rule"]
    assert two_decimals(era) == [1.10, 0.94, 1.29]
    ph_stratified = proportional_hazard_test(stratified, by_region, time_transform="rank")
    assert (ph_stratified.summary["p"] > 0.05).all()


def test_finding_6_reasons(senators, before):
    assert (before["reason"] == "Retirement").sum() == 0
    assert round(100 * (before["reason"] == "Death").mean()) == 76
    shares = estimates.reason_shares(senators).round(0)
    assert shares.loc["June 1965–1989", ["Retirement", "Death", "Resignation"]].tolist() == [55, 23, 20]
    assert shares.loc["1940–May 1965", "Resignation"] == 41

    resigned = senators[(senators["period"] == "1940–May 1965") & (senators["reason"] == "Resignation")]
    assert len(resigned) == 62
    assert (resigned["end"] >= prep.RETIREMENT_RULE_DATE).sum() == 59
