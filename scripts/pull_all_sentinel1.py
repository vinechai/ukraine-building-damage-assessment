# runs the sentinel-1 extraction for all 19 ukrainian cities in the benchmark, one
# after another. skips a city if its csv already exists, so it's safe to re-run
# if it gets interrupted partway through.
#
# usage:
#   python scripts/pull_all_sentinel1.py --project buildings-damage

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ee

from src.data.benchmark_labels import load_city
from src.data.sentinel1_gee import extract_time_series
from pull_sentinel1_timeseries import WAR_START, INFERENCE_START


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("data/sentinel1"))
    args = parser.parse_args()

    ee.Initialize(project=args.project)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cities = list(INFERENCE_START.keys())
    print(f"{len(cities)} cities queued: {cities}")

    for i, city_key in enumerate(cities):
        out_path = args.out_dir / f"{city_key}_timeseries.csv"
        if out_path.exists():
            print(f"[{i+1}/{len(cities)}] {city_key}: already done, skipping")
            continue

        print(f"\n[{i+1}/{len(cities)}] {city_key}: starting")
        try:
            gdf = load_city(city_key)
            t0 = time.time()
            df = extract_time_series(gdf, war_start=WAR_START, inference_start=INFERENCE_START[city_key])
            df.to_csv(out_path, index=False)
            elapsed = time.time() - t0
            print(f"[{i+1}/{len(cities)}] {city_key}: done in {elapsed:.0f}s, "
                  f"{len(df)} rows, {out_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"[{i+1}/{len(cities)}] {city_key}: FAILED - {e}")
            continue

    print("\nall cities processed")


if __name__ == "__main__":
    main()
