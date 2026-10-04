# pulls a recent, current sentinel-1 window for a city (not the 2022 war-period window)
# so we can see what the trained pipeline says about "now", clearly unvalidated, since
# there's no unosat assessment for the present day to check it against.
#
# reuses the exact same extraction function as the original 2022 pull, just pointed
# at a recent date range instead. war_start/inference_start are both set to the start
# of the recent window with pre_months=0, so it only fetches that window, nothing else.
#
# usage:
#   python scripts/pull_current_snapshot.py --city rubizhne --project buildings-damage --start 2026-07-01 --months 3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee
import pandas as pd

from src.data.benchmark_labels import load_city
from src.data.sentinel1_gee import extract_time_series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--start", required=True, help="start of the recent window, e.g. 2026-07-01")
    parser.add_argument("--months", type=int, default=3, help="how many months of recent data to pull")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ee.Initialize(project=args.project)
    gdf = load_city(args.city)

    # force the same orbit the original baseline used for this city. picking a
    # fresh "best orbit" independently for this new window can silently choose a
    # different, worse one (confirmed on mariupol: 95% of readings came back empty
    # on the mismatched orbit), and mixing orbits between two periods you're
    # comparing is wrong regardless of coverage.
    original_path = Path(f"data/sentinel1/{args.city.lower()}_timeseries.csv")
    if not original_path.exists():
        raise FileNotFoundError(f"no original baseline pull found for {args.city} at {original_path}, "
                                 f"needed to know which orbit to reuse")
    orbit = int(pd.read_csv(original_path, usecols=["orbit"])["orbit"].iloc[0])

    print(f"{args.city}: {len(gdf)} buildings, pulling current snapshot from {args.start} "
          f"for {args.months} month(s), forcing orbit {orbit} to match the original baseline...")

    df = extract_time_series(gdf, war_start=args.start, pre_months=0,
                              inference_start=args.start, post_months=args.months, orbit=orbit)

    out_path = args.out or Path(f"data/sentinel1/{args.city.lower()}_current_timeseries.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"done. {len(df)} rows, {df['date'].nunique()} dates, saved to {out_path}")


if __name__ == "__main__":
    main()
