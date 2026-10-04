# builds the full per-building feature table across all 19 cities and saves it.
# this is what the model trains and evaluates on.
#
# usage:
#   python scripts/build_training_table.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.build_features import build_all_features
from pull_sentinel1_timeseries import WAR_START, INFERENCE_START


def main():
    df = build_all_features(INFERENCE_START, war_start=WAR_START)
    out_path = Path("data/features.csv")
    df.to_csv(out_path, index=False)
    print(f"{len(df)} buildings, {df['city'].nunique()} cities, saved to {out_path}")
    print(df.groupby("city")["damaged"].agg(["count", "mean"]))


if __name__ == "__main__":
    main()
