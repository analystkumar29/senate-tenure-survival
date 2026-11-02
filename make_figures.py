"""Rebuild every chart in figures/ without Jupyter, and print the headline numbers.

    python make_figures.py
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from src import estimates, plots, prep

FIGURES = Path(__file__).parent / "figures"


def main() -> None:
    matplotlib.use("Agg")  # write files only; never try to open a window
    raw = prep.load_raw()
    senators = prep.prepare(raw)
    comparison = estimates.compare_methods(
        {
            "All appointments": senators,
            prep.ERA_BEFORE: senators[~senators["after_1965_rule"]],
            prep.ERA_AFTER: senators[senators["after_1965_rule"]],
        }
    )
    back = estimates.backtest(senators, cutoff="1990-01-01")
    cph = estimates.fit_cox(estimates.cox_design(senators))

    plots.use_style()
    figures = {
        "km_overall": plots.plot_km_overall(senators),
        "naive_vs_km": plots.plot_naive_vs_km(comparison, back),
        "km_by_era": plots.plot_km_by_era(senators),
        "median_by_period": plots.plot_median_by_period(senators),
        "cox_hazard_ratios": plots.plot_cox_hazard_ratios(cph),
        "reasons_by_period": plots.plot_reasons_by_period(senators),
    }
    FIGURES.mkdir(exist_ok=True)
    for name, fig in figures.items():
        fig.savefig(FIGURES / f"{name}.png", dpi=200)
        plt.close(fig)
        print(f"wrote figures/{name}.png")

    print(f"\nData stops on {prep.censoring_date(raw):%d %B %Y}; "
          f"{len(senators)} appointments, {int((~senators['observed']).sum())} still serving.")
    print("\nMedian term in years, by method:")
    print(comparison.round(1).to_string())
    print(f"\nBack-test as of {back['cutoff']:%d %b %Y} ({back['n']} appointments, "
          f"{back['still_serving_at_cutoff']} still serving then):")
    print(f"  completed only {back['completed_median']:.1f}, as if ended {back['as_ended_median']:.1f}, "
          f"Kaplan–Meier {back['km_median']:.1f} ({back['km_lower']:.1f}–{back['km_upper']:.1f}); "
          f"actual {back['eventual_median']:.1f}")


if __name__ == "__main__":
    main()
