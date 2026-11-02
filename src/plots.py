"""The charts, as functions that return a matplotlib Figure.

Both analysis.ipynb and make_figures.py call these, so the notebook and the PNGs in
figures/ always come from the same code.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test, multivariate_logrank_test
from matplotlib.ticker import PercentFormatter
from matplotlib.transforms import blended_transform_factory

from src import prep
from src.estimates import hazard_ratio_table, km_median, km_medians_by, reason_shares

# Colours: one blue for the main series, orange as the second, aqua as the third, and
# greys for anything that should recede (naive estimates, senators still serving).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NAIVE_GREY = "#b4b2a9"
STILL_SERVING_GREY = "#dcdbd4"

ERA_COLOURS = {prep.ERA_BEFORE: BLUE, prep.ERA_AFTER: ORANGE}
REASON_COLOURS = {
    "Death": BLUE,
    "Resignation": ORANGE,
    "Retirement": AQUA,
    prep.STILL_SERVING: STILL_SERVING_GREY,
}


def use_style() -> None:
    """Quiet defaults: light surface, hairline grid, no top/right frame."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SOFT,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_SOFT,
            "ytick.labelcolor": INK_SOFT,
            "legend.frameon": False,
        }
    )


def _titles(fig, title: str, subtitle: str, top: float = 0.97) -> None:
    """Left-aligned headline (the finding) with a smaller line of detail under it."""
    fig.text(0.03, top, title, ha="left", va="top", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.03, top - 0.06, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SOFT,
             linespacing=1.4)


def _km(d: pd.DataFrame, label: str) -> KaplanMeierFitter:
    return KaplanMeierFitter(label=label).fit(d["years"], event_observed=d["observed"])


def _style_at_risk_table(fig) -> None:
    """lifelines draws the at-risk counts on extra axes; make them match the chart."""
    for extra in fig.axes[1:]:
        extra.tick_params(colors=MUTED, labelcolor=INK_SOFT, labelsize=9)
        extra.grid(False)
        for spine in extra.spines.values():
            spine.set_visible(False)


def plot_km_overall(senators: pd.DataFrame):
    """Kaplan–Meier curve for every appointment, with its 95% band and the median marked."""
    kmf = _km(senators, "All appointments")
    median, lower, upper = km_median(senators["years"], senators["observed"])

    fig, ax = plt.subplots(figsize=(8, 5.4))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.24)
    kmf.plot_survival_function(ax=ax, color=BLUE, lw=2, ci_alpha=0.15, legend=False)
    ax.plot([median], [0.5], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=5)
    ax.annotate(f"Median {median:.1f} years\n(95% CI {lower:.1f}–{upper:.1f})",
                xy=(median, 0.5), xytext=(median + 3, 0.62), color=INK, fontsize=9.5,
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Years since appointment")
    ax.set_ylabel("Still in the Senate")
    add_at_risk_counts(kmf, ax=ax, rows_to_show=["At risk"], labels=["All appointments"])
    _style_at_risk_table(fig)

    n_censored = int((~senators["observed"]).sum())
    _titles(fig, "A typical Canadian senator serves about 14 years",
            f"Kaplan–Meier estimate with 95% band, {len(senators)} appointments 1867–2013. "
            f"The {n_censored} senators still\nserving when the data stopped (1 Oct 2013) "
            "count as 'served at least this long'.")
    return fig


def plot_naive_vs_km(comparison: pd.DataFrame, back: dict):
    """Median term by method, for groups with more and more senators still serving."""
    groups = [
        (f"{prep.ERA_BEFORE}  ·  none still serving", comparison.loc[prep.ERA_BEFORE]),
        (f"{prep.ERA_AFTER}  ·  {comparison.loc[prep.ERA_AFTER, 'pct_still_serving']:.0f}% "
         "still serving in 2013", comparison.loc[prep.ERA_AFTER]),
        (f"Back-test: appointed June 1965–1989, data as of 1 Jan 1990  ·  "
         f"{100 * back['still_serving_at_cutoff'] / back['n']:.0f}% still serving then",
         pd.Series(back)),
    ]
    methods = [
        ("Completed terms only", "completed_median", NAIVE_GREY),
        ("Sitting senators treated as\nleaving on the data date", "as_ended_median", NAIVE_GREY),
        ("Kaplan–Meier", "km_median", BLUE),
    ]

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    fig.subplots_adjust(left=0.3, right=0.97, top=0.8, bottom=0.08)
    # Group headers line up with the title at the figure's left edge (x in figure
    # coordinates) but sit at a data height (y in data coordinates).
    header_transform = blended_transform_factory(fig.transFigure, ax.transData)
    y, yticks, ylabels = 0.0, [], []
    for header, row in groups:
        ax.text(0.03, y, header, transform=header_transform, fontsize=9.5, fontweight="bold",
                color=INK, va="center")
        bar_ys = [y - 0.85, y - 1.85, y - 2.85]
        for bar_y, (label, key, colour) in zip(bar_ys, methods, strict=True):
            value = row[key]
            ax.barh(bar_y, value, height=0.62, color=colour)
            if key == "km_median":
                ax.errorbar(value, bar_y, xerr=[[value - row["km_lower"]], [row["km_upper"] - value]],
                            fmt="none", ecolor=INK, elinewidth=1, capsize=3)
                text = f"{value:.1f}  ({row['km_lower']:.1f}–{row['km_upper']:.1f})"
                ax.text(row["km_upper"] + 0.4, bar_y, text, va="center", fontsize=9, color=INK)
            else:
                ax.text(value + 0.4, bar_y, f"{value:.1f}", va="center", fontsize=9, color=INK_SOFT)
            yticks.append(bar_y)
            ylabels.append(label)
        y = bar_ys[-1] - 1.15  # room before the next group's header

    # The answer key exists only for the back-test: draw it across that group's bars.
    truth = back["eventual_median"]
    ax.plot([truth, truth], [bar_ys[-1] - 0.42, bar_ys[0] + 0.36], color=INK, lw=1.4)
    ax.text(truth + 0.3, bar_ys[0], f"What actually happened: {truth:.1f}", fontsize=9,
            color=INK, va="center")

    ax.set_yticks(yticks, ylabels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(bar_ys[-1] - 0.6, 0.45)
    ax.set_xlim(0, 26)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Median term (years); 95% CI in brackets")
    _titles(fig, "Ignoring senators still in office makes terms look years shorter",
            "Three ways to get a median term. The shortcuts drop sitting senators or cut "
            "their terms off at the\ndata date; Kaplan–Meier counts each one as 'at least "
            "this long'. The more still serving, the bigger the gap.")
    return fig


def plot_km_by_era(senators: pd.DataFrame):
    """Kaplan–Meier curves before vs after the 1965 retire-at-75 rule."""
    fig, ax = plt.subplots(figsize=(8, 5.6))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.78, bottom=0.26)
    fitters = []
    for era in prep.ERAS:
        d = senators[senators["era"] == era]
        kmf = _km(d, era)
        kmf.plot_survival_function(ax=ax, color=ERA_COLOURS[era], lw=2, ci_alpha=0.12)
        fitters.append(kmf)

    before = senators[~senators["after_1965_rule"]]
    after = senators[senators["after_1965_rule"]]
    test = logrank_test(before["years"], after["years"], before["observed"], after["observed"])
    m_before, _, _ = km_median(before["years"], before["observed"])
    m_after, _, _ = km_median(after["years"], after["observed"])

    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Years since appointment")
    ax.set_ylabel("Still in the Senate")
    ax.legend(loc="upper right", fontsize=9.5, labelcolor=INK_SOFT)
    add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"],
                       labels=["Before June 1965", "June 1965 or later"])
    _style_at_risk_table(fig)

    _titles(fig, "After the 1965 rule: a similar typical term, shorter longest terms",
            f"Kaplan–Meier with 95% bands. Median {m_before:.1f} years before the retire-at-75 "
            f"rule vs {m_after:.1f} after (log-rank\np = {test.p_value:.2f}). Longest term: "
            f"{before['years'].max():.1f} years before the rule, {after['years'].max():.1f} after.")
    return fig


def plot_median_by_period(senators: pd.DataFrame):
    """Kaplan–Meier median and 95% CI for each appointment period."""
    table = km_medians_by(senators, "period")
    test = multivariate_logrank_test(senators["years"], senators["period"], senators["observed"])

    fig, ax = plt.subplots(figsize=(8, 4.9))
    fig.subplots_adjust(left=0.19, right=0.72, top=0.77, bottom=0.12)
    ys = np.arange(len(table))[::-1]  # oldest period at the top
    for y, row in zip(ys, table.itertuples(), strict=True):
        ax.plot([row.km_lower, row.km_upper], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot([row.km_median], [y], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.text(row.km_median, y + 0.22, f"{row.km_median:.1f}", ha="center", va="bottom",
                fontsize=9, color=INK)
        ax.text(1.03, y, f"n = {row.n} · {row.pct_still_serving:.0f}% still serving",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK_SOFT)

    # Half a row above "June 1965–1989", the first period under the retire-at-75 rule.
    rule_y = ys[list(table.index).index("June 1965–1989")] + 0.5
    ax.axhline(rule_y, color=INK_SOFT, lw=0.9)
    ax.text(18.8, rule_y + 0.1, "Retire at 75 from June 1965 ↓", ha="right", va="bottom",
            fontsize=8.5, color=INK_SOFT)

    ax.set_yticks(ys, table.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(8, 19)
    ax.set_ylim(-0.6, len(table) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Kaplan–Meier median term (years), with 95% CI")
    low, high = table["km_median"].min(), table["km_median"].max()
    _titles(fig, f"Every period's typical term falls between {low:.1f} and {high:.1f} years",
            "Senators grouped by when they were appointed. A log-rank test across the six "
            f"periods finds no\nclear difference (p = {test.p_value:.2f}). The 1990–2013 "
            "estimate is the least certain: half that group was still serving.")
    return fig


COX_LABELS = {
    "appointed after 1965 rule": "Appointed June 1965 or later\n(vs before)",
    "party: Conservative": "Conservative (vs Liberal)",
    "party: Other": "Other party or independent\n(vs Liberal)",
    "region: Atlantic": "Atlantic (vs Ontario)",
    "region: Quebec": "Quebec (vs Ontario)",
    "region: West & North": "West & North (vs Ontario)",
}


def plot_cox_hazard_ratios(cph):
    """Forest plot of the Cox model's hazard ratios."""
    table = hazard_ratio_table(cph).loc[list(COX_LABELS)]
    fig, ax = plt.subplots(figsize=(8, 4.9))
    fig.subplots_adjust(left=0.29, right=0.74, top=0.75, bottom=0.13)
    ys = np.arange(len(table))[::-1]
    ax.axvline(1, color=INK_SOFT, lw=0.9)
    for y, (_, row) in zip(ys, table.iterrows(), strict=True):
        ax.plot([row["lower_95"], row["upper_95"]], [y, y], color=BLUE, lw=2, solid_capstyle="round")
        ax.plot([row["hazard_ratio"]], [y], "o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.text(1.03, y, f"{row['hazard_ratio']:.2f}  ({row['lower_95']:.2f}–{row['upper_95']:.2f})",
                transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK)
    ax.text(1.03, len(table) - 0.35, "Hazard ratio (95% CI)", transform=ax.get_yaxis_transform(),
            fontsize=9, color=INK_SOFT)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.45, 2.2)
    ax.set_yticks(ys, [COX_LABELS[name] for name in table.index])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(table) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Hazard ratio (log scale): below 1 = tends to stay longer, above 1 = leaves sooner")
    _titles(fig, "Era, party and region barely change how fast senators leave",
            "Cox proportional-hazards model with era, party and region fitted together. Every "
            f"interval crosses 1,\nand the concordance is {cph.concordance_index_:.2f} (0.5 = coin "
            "flip): these facts say little about any one senator's term.")
    return fig


def plot_reasons_by_period(senators: pd.DataFrame):
    """100% stacked bars: how each appointment period's terms ended."""
    shares = reason_shares(senators)
    fig, ax = plt.subplots(figsize=(8, 5.0))
    fig.subplots_adjust(left=0.19, right=0.97, top=0.71, bottom=0.11)
    ys = np.arange(len(shares))[::-1]
    left = np.zeros(len(shares))
    for reason in shares.columns:
        values = shares[reason].to_numpy()
        # The white edge leaves a thin gap between segments instead of a drawn border.
        ax.barh(ys, values, left=left, height=0.62, color=REASON_COLOURS[reason],
                edgecolor=SURFACE, linewidth=1.5, label=reason)
        for y, x0, v in zip(ys, left, values, strict=True):
            if v >= 10:  # label only segments wide enough to hold the text
                colour = INK if reason == prep.STILL_SERVING else "white"
                ax.text(x0 + v / 2, y, f"{v:.0f}%", ha="center", va="center", fontsize=9,
                        color=colour)
        left += values

    # Half a row above "June 1965–1989", the first period under the retire-at-75 rule.
    rule_y = ys[list(shares.index).index("June 1965–1989")] + 0.5
    ax.axhline(rule_y, color=INK_SOFT, lw=0.9, xmin=-0.22, clip_on=False)
    ax.set_yticks(ys, shares.index)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(shares) - 0.5)
    ax.xaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of appointments in the period")
    ax.legend(ncols=4, loc="lower left", bbox_to_anchor=(0, 1.01), fontsize=9.5,
              labelcolor=INK_SOFT, handlelength=1.2, columnspacing=1.5)

    before = senators[~senators["after_1965_rule"]]
    pct_death = 100 * (before["reason"] == "Death").mean()
    _titles(fig, "After the 1965 rule, retirement replaced death as the usual way out",
            "How each period's terms ended, as of 1 Oct 2013. Of the senators appointed before "
            f"June 1965,\n{pct_death:.0f}% died in office; the data records no retirements "
            "among them. The line marks the 1965 rule.")
    return fig
