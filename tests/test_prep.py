import pandas as pd
import pytest

from src import prep


def test_days_to_years():
    assert prep.days_to_years(365.25) == 1.0
    assert prep.days_to_years(pd.Series([0, 730.5])).tolist() == [0.0, 2.0]


@pytest.mark.parametrize(
    "start, expected",
    [
        ("1867-10-23", prep.ERA_BEFORE),
        ("1965-06-01", prep.ERA_BEFORE),  # the day before the rule came into force
        ("1965-06-02", prep.ERA_AFTER),   # the rule applies from this day on
        ("2013-03-25", prep.ERA_AFTER),
    ],
)
def test_assign_era_splits_on_the_1965_rule(start, expected):
    era = prep.assign_era(pd.Series(pd.to_datetime([start])))
    assert era.iloc[0] == expected


@pytest.mark.parametrize(
    "start, expected",
    [
        ("1867-10-23", "1867–1889"),
        ("1889-12-31", "1867–1889"),
        ("1890-01-01", "1890–1914"),
        ("1939-12-31", "1915–1939"),
        ("1965-06-01", "1940–May 1965"),
        ("1965-06-02", "June 1965–1989"),
        ("1990-01-01", "1990–2013"),
        ("2013-03-25", "1990–2013"),
    ],
)
def test_assign_period_boundaries(start, expected):
    period = prep.assign_period(pd.Series(pd.to_datetime([start])))
    assert period.iloc[0] == expected


def test_periods_and_eras_agree():
    # A period never straddles the 1965 rule, so each period sits in exactly one era.
    starts = pd.Series(pd.date_range("1867-10-23", "2013-03-25", freq="7D"))
    pairs = pd.DataFrame({"period": prep.assign_period(starts), "era": prep.assign_era(starts)})
    assert pairs.groupby("period", observed=True)["era"].nunique().eq(1).all()


@pytest.mark.parametrize(
    "label, family",
    [
        ("Liberal Party of Canada", "Liberal"),
        ("Independent Liberal", "Liberal"),
        ("Unionist (Liberal)", "Liberal"),
        ("Conservative (1867-1942)", "Conservative"),
        ("Liberal-Conservative", "Conservative"),  # "Conservative" wins over "Liberal"
        ("Progressive Conservative Party", "Conservative"),
        ("Conservative Party of Canada", "Conservative"),
        ("Independent", "Other"),
        ("New Democratic Party", "Other"),
        ("Reform Party", "Other"),
    ],
)
def test_party_family(label, family):
    assert prep.party_family(label) == family


@pytest.mark.parametrize(
    "province, expected",
    [
        ("Newfoundland and Labrador", "Atlantic"),
        ("Maritimes (Division)", "Atlantic"),
        ("Quebec (Division)", "Quebec"),
        ("Ontario", "Ontario"),
        ("British Columbia", "West & North"),
        ("Nunavut", "West & North"),
        ("Western Provinces (Division)", "West & North"),
    ],
)
def test_region(province, expected):
    assert prep.region(province) == expected


def test_region_rejects_unknown_names():
    with pytest.raises(KeyError):
        prep.region("Atlantis")


def _toy_senators():
    starts = pd.to_datetime(["1970-01-01", "1980-01-01", "1995-01-01"])
    ends = pd.to_datetime(["1985-01-01", "2000-01-01", "2013-10-01"])
    return pd.DataFrame(
        {
            "start": starts,
            "end": ends,
            "observed": [True, True, False],
            "reason": ["Death", "Retirement", prep.STILL_SERVING],
            "years": prep.days_to_years((ends - starts).days),
        }
    )


def test_censor_at_rebuilds_an_earlier_snapshot():
    toy = _toy_senators()
    snapshot = prep.censor_at(toy, "1990-01-01")

    # The 1995 appointment hadn't happened yet in 1990.
    assert len(snapshot) == 2
    # The 1970 term had already ended, so it is unchanged.
    assert bool(snapshot.iloc[0]["observed"]) is True
    assert snapshot.iloc[0]["years"] == pytest.approx(15.0, abs=0.01)
    # The 1980 term was still running on 1 Jan 1990: censored at 10 years.
    assert bool(snapshot.iloc[1]["observed"]) is False
    assert snapshot.iloc[1]["reason"] == prep.STILL_SERVING
    assert snapshot.iloc[1]["years"] == pytest.approx(10.0, abs=0.01)


def test_censor_at_leaves_the_input_alone():
    toy = _toy_senators()
    before = toy.copy()
    prep.censor_at(toy, "1990-01-01")
    pd.testing.assert_frame_equal(toy, before)


# The real data ships with lifelines, so these run offline in a second or two.


@pytest.fixture(scope="module")
def raw():
    return prep.load_raw()


@pytest.fixture(scope="module")
def senators(raw):
    return prep.prepare(raw)


def test_censoring_date_is_1_october_2013(raw):
    assert prep.censoring_date(raw) == pd.Timestamp("2013-10-01")


def test_prepare_keeps_every_term_that_was_served(raw, senators):
    assert len(raw) == 933
    assert len(senators) == 930  # 3 declined appointments removed
    assert int(senators["observed"].sum()) == 831
    assert int((~senators["observed"]).sum()) == 99
    assert (senators.loc[~senators["observed"], "end"] == pd.Timestamp("2013-10-01")).all()


def test_prepare_durations_match_the_raw_day_counts(raw, senators):
    served = raw[raw["reason"] != "Appointment declined"].reset_index(drop=True)
    assert (senators["years"] * prep.DAYS_PER_YEAR).round(6).tolist() == served["diff_days"].tolist()


def test_prepare_assigns_every_row_to_a_group(senators):
    for column in ["era", "period", "party_family", "region"]:
        assert senators[column].notna().all(), column
    assert set(senators["region"]) == set(prep.REGIONS)
    assert set(senators["party_family"]) == set(prep.PARTY_FAMILIES)
