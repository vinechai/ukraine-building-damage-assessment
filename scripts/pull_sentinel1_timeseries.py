# pulls raw sentinel-1 time series for one city's buildings and saves it to a csv.
# needs earth engine set up first (see readme).
#
# usage:
#   python scripts/pull_sentinel1_timeseries.py --city Bucha --project buildings-damage

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee

from src.data.benchmark_labels import load_city
from src.data.sentinel1_gee import extract_time_series

# war start pwtt used for all ukrainian cities. inference_start is per city,
# it's the date pwtt used, which matches when unosat did the damage check there.
WAR_START = "2022-02-22"

# inference_start per city, taken from pwtt's own eval script, matches the date
# already in each city's csv filename
INFERENCE_START = {
    "bucha": "2022-03-31",
    "irpin": "2022-03-31",
    "hostomel": "2022-03-31",
    "chernihiv": "2022-03-22",
    "mariupol": "2022-05-12",
    "kharkiv": "2022-06-15",
    "makariv": "2022-03-16",
    "trostianets": "2022-03-25",
    "okhtyrka": "2022-03-25",
    "sumy": "2022-07-08",
    "shchastia": "2022-07-07",
    "rubizhne": "2022-07-09",
    "sievierodonetsk": "2022-07-27",
    "mykolaiv": "2022-07-21",
    "kremenchuk": "2022-06-29",
    "kramatorsk": "2022-07-24",
    "melitopol": "2022-08-02",
    "avdiivka": "2022-09-20",
    "lysychansk": "2022-09-21",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--project", required=True, help="your earth engine cloud project id")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    city_key = args.city.lower()
    if city_key not in INFERENCE_START:
        print(f"no inference_start known for {args.city}, add it to INFERENCE_START in this script")
        return

    gdf = load_city(args.city)
    print(f"{args.city}: {len(gdf)} buildings, pulling sentinel-1 time series...")

    t0 = time.time()
    df = extract_time_series(gdf, war_start=WAR_START, inference_start=INFERENCE_START[city_key])
    elapsed = time.time() - t0

    out_path = args.out or Path(f"data/sentinel1/{args.city.lower()}_timeseries.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"done in {elapsed:.0f}s. {len(df)} rows ({df['building_id'].nunique()} buildings x "
          f"{df['date'].nunique()} dates). saved to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
