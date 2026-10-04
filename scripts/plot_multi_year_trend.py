# looks at the raw signal (not the model's prediction) across several points in time,
# to check whether a city's sentinel-1 readings show a sudden jump around a real,
# documented strike date, or just drift smoothly/randomly. a real jump lining up with a
# real event is actual evidence; a smooth drift with no particular jump points back
# toward pipeline/calibration noise rather than genuine new damage.
#
# usage:
#   python scripts/plot_multi_year_trend.py --city sumy --periods "2023 baseline:sumy_2023baseline" "pre nov 2024 strike:sumy_2024nov" "pre apr 2025 strike:sumy_2025apr" "2026 current:sumy_current"

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.data.build_features import _window_stats

WAR_START = "2022-02-22"


def period_summary(city: str, label: str, suffix: str) -> dict:
    pre = pd.read_csv(f"data/sentinel1/{city}_timeseries.csv", parse_dates=["date"])
    pre = pre[pre["date"] < pd.Timestamp(WAR_START)]
    pre_stats = _window_stats(pre, "pre")

    period = pd.read_csv(f"data/sentinel1/{city}_{suffix}_timeseries.csv", parse_dates=["date"])
    period_stats = _window_stats(period, "cur")

    merged = pre_stats.join(period_stats, how="inner")
    diff_vv = merged["cur_vv_mean"] - merged["pre_vv_mean"]
    z_vv = diff_vv / merged["pre_vv_std"].replace(0, np.nan)
    diff_vh = merged["cur_vh_mean"] - merged["pre_vh_mean"]
    z_vh = diff_vh / merged["pre_vh_std"].replace(0, np.nan)
    combined_z = np.sqrt(z_vv ** 2 + z_vh ** 2)

    return {
        "period": label,
        "dates": f"{period['date'].min():%Y-%m-%d} to {period['date'].max():%Y-%m-%d}",
        "n_passes": period["date"].nunique(),
        "mean_diff_vv": diff_vv.mean(),
        "mean_combined_z": combined_z.mean(),
        "pct_combined_z_over_2": (combined_z > 2).mean() * 100,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--periods", nargs="+", required=True,
                         help='each as "label:file_suffix", matching data/sentinel1/<city>_<suffix>_timeseries.csv')
    args = parser.parse_args()

    rows = []
    for spec in args.periods:
        label, suffix = spec.split(":")
        rows.append(period_summary(args.city, label, suffix))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
