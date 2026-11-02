"""Load the Canadian senators data and turn it into one tidy row per appointment.

Each function is one decision I can point to: how a term is measured, which era an
appointment belongs to, and how party labels and provinces are grouped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines.datasets import load_canadian_senators

DAYS_PER_YEAR = 365.25  # average calendar year, counting leap years

# Constitution Act, 1965 (in force 2 June 1965): senators summoned after it came into
# force must retire at 75. Senators appointed before it kept their life appointments.
RETIREMENT_RULE_DATE = pd.Timestamp("1965-06-02")

ERA_BEFORE = "Appointed before June 1965"
ERA_AFTER = "Appointed June 1965 or later"
ERAS = [ERA_BEFORE, ERA_AFTER]

# Roughly 25-year periods. One boundary sits exactly on the 1965 rule so that no period
# mixes life appointments with retire-at-75 appointments.
PERIOD_EDGES = pd.to_datetime(
    ["1867-01-01", "1890-01-01", "1915-01-01", "1940-01-01",
     "1965-06-02", "1990-01-01", "2014-01-01"]
)
PERIOD_LABELS = [
    "1867–1889", "1890–1914", "1915–1939",
    "1940–May 1965", "June 1965–1989", "1990–2013",
]

REGION_BY_PROVINCE = {
    "Newfoundland and Labrador": "Atlantic",
    "Nova Scotia": "Atlantic",
    "New Brunswick": "Atlantic",
    "Prince Edward Island": "Atlantic",
    "Quebec": "Quebec",
    "Ontario": "Ontario",
    "Manitoba": "West & North",
    "Saskatchewan": "West & North",
    "Alberta": "West & North",
    "British Columbia": "West & North",
    # Only 11 appointments came from the territories: too few to estimate on their
    # own, so they share a group with the western provinces.
    "Yukon": "West & North",
    "Northwest Territories": "West & North",
    "Nunavut": "West & North",
    # Eight senators appointed on 27 Sep 1990 are listed by Senate division
    # rather than by province. Each division maps cleanly onto one region.
    "Maritimes (Division)": "Atlantic",
    "Quebec (Division)": "Quebec",
    "Ontario (Division)": "Ontario",
    "Western Provinces (Division)": "West & North",
}
REGIONS = ["Atlantic", "Quebec", "Ontario", "West & North"]

PARTY_FAMILIES = ["Liberal", "Conservative", "Other"]

STILL_SERVING = "Still serving"


def load_raw() -> pd.DataFrame:
    """The dataset exactly as lifelines ships it (933 rows, one per appointment)."""
    return load_canadian_senators()


def censoring_date(raw: pd.DataFrame) -> pd.Timestamp:
    """The date the data stops: every sitting senator's term is cut off here.

    In the raw file, sitting senators share one end_date with a time of day attached,
    which looks like the moment the data was collected. I keep only the date.
    """
    ends = pd.to_datetime(raw.loc[~raw["observed"], "end_date"], format="ISO8601")
    dates = ends.dt.normalize().unique()
    if len(dates) != 1:
        raise ValueError(f"expected one censoring date, found {len(dates)}")
    return pd.Timestamp(dates[0])


def days_to_years(days):
    """Convert a number of days (scalar or Series) into years."""
    return days / DAYS_PER_YEAR


def assign_era(start: pd.Series) -> pd.Series:
    """Label each appointment as before or after the 1965 retire-at-75 rule."""
    labels = np.where(start >= RETIREMENT_RULE_DATE, ERA_AFTER, ERA_BEFORE)
    return pd.Series(pd.Categorical(labels, categories=ERAS, ordered=True), index=start.index)


def assign_period(start: pd.Series) -> pd.Series:
    """Label each appointment with its roughly 25-year appointment period.

    right=False makes each period include its own start date, so an appointment on
    2 June 1965 would fall in "June 1965–1989", the same side as the rule.
    """
    return pd.cut(start, bins=PERIOD_EDGES, right=False, labels=PERIOD_LABELS)


def party_family(affiliation: str) -> str:
    """Group the 17 party labels in the data into three families, by name.

    "Conservative" is checked first so that "Liberal-Conservative" and "Progressive
    Conservative" both count as Conservative. Every Liberal-Conservative appointment in
    the data came from the same sources (Macdonald, Abbott, Bowell, or the 1867
    proclamation) that made "Conservative (1867-1942)" appointments.
    """
    if "Conservative" in affiliation:
        return "Conservative"
    if "Liberal" in affiliation:
        return "Liberal"
    return "Other"  # independents and small parties


def region(province: str) -> str:
    """Map a province, territory or Senate division to one of four regions.

    Raises KeyError for a name it doesn't know, so a surprise in the data fails loudly
    instead of silently becoming a missing value.
    """
    return REGION_BY_PROVINCE[province]


def prepare(raw: pd.DataFrame) -> pd.DataFrame:
    """One tidy row per appointment, with the duration in years and the groupings."""
    # Three people named at Confederation declined the appointment (0 days, never sat).
    # They are not terms in the Senate, so they don't belong in a tenure analysis.
    served = raw[raw["reason"] != "Appointment declined"]

    start = pd.to_datetime(served["start_date"], format="ISO8601")
    # Sitting senators' end_date carries a time of day; drop it so every date is a day.
    end = pd.to_datetime(served["end_date"], format="ISO8601").dt.normalize()

    tidy = pd.DataFrame(
        {
            "name": served["Name"],
            "start": start,
            "end": end,
            # observed=True: the term has ended. False: still serving when the data
            # stopped, so we only know the term is at least this long (right-censored).
            "observed": served["observed"].astype(bool),
            "reason": served["reason"].fillna(STILL_SERVING),
            "years": days_to_years((end - start).dt.days),
            "after_1965_rule": start >= RETIREMENT_RULE_DATE,
            "era": assign_era(start),
            "period": assign_period(start),
            "party_family": served["Political Affiliation at Appointment"].map(party_family),
            "region": served["Province / Territory"].map(region),
        }
    )
    return tidy.reset_index(drop=True)


def censor_at(senators: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Rebuild the data as it would have looked if it had been collected on `cutoff`.

    This is for a back-test: censor the data at a past date, estimate from that
    snapshot, then compare with what we now know actually happened.
    """
    cutoff = pd.Timestamp(cutoff)
    # Appointments after the cutoff hadn't happened yet, so they can't be in the snapshot.
    snapshot = senators[senators["start"] < cutoff].copy()
    # Anyone whose term ended after the cutoff was still serving on that day.
    not_ended_yet = snapshot["end"] > cutoff
    snapshot.loc[not_ended_yet, "end"] = cutoff
    snapshot.loc[not_ended_yet, "observed"] = False
    snapshot.loc[not_ended_yet, "reason"] = STILL_SERVING
    snapshot["years"] = days_to_years((snapshot["end"] - snapshot["start"]).dt.days)
    return snapshot
