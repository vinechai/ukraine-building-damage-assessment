# turns the raw per-date VV/VH readings into one row per building: "before" stats and
# two versions of "after" stats, computed by us straight from the raw sentinel-1 data
# (not copied from pwtt's own columns). this is the actual training table for our model.
#
# two "after" windows, because they capture different things:
#   - narrow: just the month right around when unosat did their assessment
#   - wide: the whole period from when the war reached that area up to the assessment
#     (this can be several months, e.g. mariupol's siege ran feb-may 2022 but the
#     narrow window only covers the final month before assessment. the wide window
#     catches damage that happened earlier in that stretch too)
#
# also, instead of only looking at the average signal, we track the single most
# extreme date in the post period (max_z_vv / max_z_vh). pwtt's own method does the
# same thing (their "max_change") for a good reason: a mean can wash out a real,
# sudden one-day jump if it gets averaged together with normal days.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_labels import load_city

SENTINEL1_DIR = Path("data/sentinel1")


def _window_stats(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return df.groupby("building_id").agg(**{
        f"{prefix}_vv_mean": ("VV", "mean"), f"{prefix}_vv_std": ("VV", "std"),
        f"{prefix}_vv_min": ("VV", "min"), f"{prefix}_vv_max": ("VV", "max"),
        f"{prefix}_vh_mean": ("VH", "mean"), f"{prefix}_vh_std": ("VH", "std"),
        f"{prefix}_vh_min": ("VH", "min"), f"{prefix}_vh_max": ("VH", "max"),
        f"n_{prefix}": ("VV", "count"),
    })


def build_city_features(city: str, war_start: str, inference_start: str, post_months: int = 1) -> pd.DataFrame:
    ts_path = SENTINEL1_DIR / f"{city.lower()}_timeseries.csv"
    ts = pd.read_csv(ts_path, parse_dates=["date"])

    war_start = pd.Timestamp(war_start)
    inference_start = pd.Timestamp(inference_start)
    post_end = inference_start + pd.DateOffset(months=post_months)

    pre = ts[ts["date"] < war_start]
    post_narrow = ts[(ts["date"] >= inference_start) & (ts["date"] < post_end)]
    post_wide = ts[(ts["date"] >= war_start) & (ts["date"] < post_end)]

    pre_stats = _window_stats(pre, "pre")
    narrow_stats = _window_stats(post_narrow, "post")
    wide_stats = _window_stats(post_wide, "wide")

    features = pre_stats.join(narrow_stats, how="inner").join(wide_stats, how="inner")

    # a building with only one satellite pass in a window has an undefined standard
    # deviation (need at least 2 points), which would otherwise get it dropped
    # entirely later on. one observation genuinely has no measured variability, so
    # treating that as 0 is the right call, not a workaround.
    std_cols = [c for c in features.columns if c.endswith("_std")]
    features[std_cols] = features[std_cols].fillna(0)

    for window, label in [("post", ""), ("wide", "_wide")]:
        features[f"diff_vv{label}"] = features[f"{window}_vv_mean"] - features["pre_vv_mean"]
        features[f"diff_vh{label}"] = features[f"{window}_vh_mean"] - features["pre_vh_mean"]
        features[f"z_vv{label}"] = features[f"diff_vv{label}"] / features["pre_vv_std"].replace(0, np.nan)
        features[f"z_vh{label}"] = features[f"diff_vh{label}"] / features["pre_vh_std"].replace(0, np.nan)
        features[f"combined_z{label}"] = np.sqrt(features[f"z_vv{label}"] ** 2 + features[f"z_vh{label}"] ** 2)

    # single most extreme date in each post window, in pre-war standard deviations.
    # joined back to pre_stats first so every reading can be turned into a z-score.
    pre_lookup = pre_stats[["pre_vv_mean", "pre_vv_std", "pre_vh_mean", "pre_vh_std"]].reset_index()

    def max_abs_z(df: pd.DataFrame, colname: str) -> pd.Series:
        merged = df.merge(pre_lookup, on="building_id", how="inner")
        z_vv = (merged["VV"] - merged["pre_vv_mean"]) / merged["pre_vv_std"].replace(0, np.nan)
        z_vh = (merged["VH"] - merged["pre_vh_mean"]) / merged["pre_vh_std"].replace(0, np.nan)
        merged[colname + "_vv"] = z_vv.abs()
        merged[colname + "_vh"] = z_vh.abs()
        return merged.groupby("building_id")[[colname + "_vv", colname + "_vh"]].max()

    features = features.join(max_abs_z(post_narrow, "max_z"), how="left")
    features = features.join(max_abs_z(post_wide, "max_z_wide"), how="left")

    # pwtt's own files have a real data quality issue: about 2% of building ids (FID)
    # are reused across two different footprint rows (checked across all 19 cities).
    # most of the time both rows agree on the damage label anyway (probably a harmless
    # export artifact), but sometimes they genuinely disagree, meaning the id refers to
    # two different buildings that can't be told apart. we drop those ambiguous ids
    # rather than guess, and keep one row for the ids where duplicates agree.
    labels = load_city(city)[["FID", "damaged", "max_change", "area"]]
    label_agreement = labels.groupby("FID")["damaged"].nunique()
    ambiguous_ids = label_agreement[label_agreement > 1].index
    labels = labels[~labels["FID"].isin(ambiguous_ids)].drop_duplicates("FID").set_index("FID")

    features = features.join(labels, how="inner")
    features["city"] = city

    return features.reset_index().rename(columns={"index": "building_id"})


def build_snapshot_features(
    city: str, war_start: str, snapshot_ts_path: Path, narrow_passes: int = 2, wide_passes: int | None = None
) -> pd.DataFrame:
    """features for a fresh, unlabeled snapshot (e.g. "what does the signal look like
    right now") instead of the original 2022 war-period comparison. reuses the exact
    same pre-war baseline as training (so the model sees the same kind of "before"
    numbers it learned from), but compares it against a recent window instead.

    there's no damage label for "now" (nobody has done a unosat-style assessment of
    the present day), so this can only ever be used for prediction, never evaluation.

    important: the "narrow" window here must use the same number of satellite passes
    that the model was trained on, not just the same calendar length. an earlier version
    of this function fed the *entire* pulled window (e.g. 3 months) into both the narrow
    and wide slots. that was a real bug, not a minor approximation.

    the standard deviation features (how much a building's signal wobbles) are the
    sensitive part, not the mean. averaging more numbers does not make a mean bigger,
    but with very few passes (1-2, common in 2022 training data) you simply cannot see
    a building's real natural variability, the spread looks small almost by mathematical
    necessity, not because the signal was actually stable. with more passes available
    (checked: averaged ~0.04 std across training, vs ~1.3 using every pass in a 3-month
    snapshot, over 30x larger, on kremenchuk) you see a bigger, probably more honest
    spread, but one the model never learned to interpret since it never saw numbers that
    size during training. its predicted probabilities collapsed to a near-random ~0.5
    for most buildings, not because of a genuine years-later extrapolation limit, but
    because of this mismatch in how many passes went into the same-named feature.

    first fix attempt used "last 1 calendar month" for the narrow window, which helped
    but didn't fully close the gap. checked training data directly and most cities'
    narrow window had only 2 to 3 actual satellite passes (n_post), not because of the
    1-month cutoff itself but because sentinel-1's revisit cadence in 2022 was sparser
    than it is now (likely sentinel-1c joining the constellation since). a "last 1
    month" slice of today's denser coverage still pulls in far more passes (6, checked
    on kremenchuk) than training ever saw. so this matches by pass *count* instead of
    calendar time, the last `narrow_passes` satellite passes, whatever calendar span
    that covers, which is the actually comparable quantity, not the date window.

    the "wide" window has the exact same shaped bug. training's wide window also had a
    fixed, usually larger pass count (about 12 for kremenchuk), and the snapshot was
    feeding it every pass in the pulled period (16 or more, coverage being denser now).
    fixed the same way: slice to `wide_passes` most recent passes. if not given, falls
    back to using everything pulled (the old, buggy behavior). callers should look up
    each city's own real training wide pass count and pass it explicitly.
    """
    original_ts = pd.read_csv(SENTINEL1_DIR / f"{city.lower()}_timeseries.csv", parse_dates=["date"])
    pre = original_ts[original_ts["date"] < pd.Timestamp(war_start)]
    pre_stats = _window_stats(pre, "pre")

    snapshot_ts_full = pd.read_csv(snapshot_ts_path, parse_dates=["date"])
    all_dates = sorted(snapshot_ts_full["date"].unique())

    recent_dates = all_dates[-narrow_passes:]
    snapshot_narrow = snapshot_ts_full[snapshot_ts_full["date"].isin(recent_dates)]

    wide_dates = all_dates[-wide_passes:] if wide_passes is not None else all_dates
    snapshot_ts = snapshot_ts_full[snapshot_ts_full["date"].isin(wide_dates)]

    narrow_stats = _window_stats(snapshot_narrow, "post")
    wide_stats = _window_stats(snapshot_ts, "wide")

    features = pre_stats.join(narrow_stats, how="inner").join(wide_stats, how="inner")
    std_cols = [c for c in features.columns if c.endswith("_std")]
    features[std_cols] = features[std_cols].fillna(0)

    for window, label in [("post", ""), ("wide", "_wide")]:
        features[f"diff_vv{label}"] = features[f"{window}_vv_mean"] - features["pre_vv_mean"]
        features[f"diff_vh{label}"] = features[f"{window}_vh_mean"] - features["pre_vh_mean"]
        features[f"z_vv{label}"] = features[f"diff_vv{label}"] / features["pre_vv_std"].replace(0, np.nan)
        features[f"z_vh{label}"] = features[f"diff_vh{label}"] / features["pre_vh_std"].replace(0, np.nan)
        features[f"combined_z{label}"] = np.sqrt(features[f"z_vv{label}"] ** 2 + features[f"z_vh{label}"] ** 2)

    pre_lookup = pre_stats[["pre_vv_mean", "pre_vv_std", "pre_vh_mean", "pre_vh_std"]].reset_index()

    def max_abs_z(df: pd.DataFrame, colname: str) -> pd.DataFrame:
        merged = df.merge(pre_lookup, on="building_id", how="inner")
        z_vv = (merged["VV"] - merged["pre_vv_mean"]) / merged["pre_vv_std"].replace(0, np.nan)
        z_vh = (merged["VH"] - merged["pre_vh_mean"]) / merged["pre_vh_std"].replace(0, np.nan)
        merged[colname + "_vv"] = z_vv.abs()
        merged[colname + "_vh"] = z_vh.abs()
        return merged.groupby("building_id")[[colname + "_vv", colname + "_vh"]].max()

    features = features.join(max_abs_z(snapshot_narrow, "max_z"), how="left")
    features = features.join(max_abs_z(snapshot_ts, "max_z_wide"), how="left")

    labels = load_city(city)[["FID", "area"]].set_index("FID")  # no damage label exists for "now"
    features = features.join(labels, how="inner")
    features["city"] = city

    return features.reset_index().rename(columns={"index": "building_id"})


def build_all_features(cities: dict[str, str], war_start: str) -> pd.DataFrame:
    tables = []
    for city, inference_start in cities.items():
        try:
            tables.append(build_city_features(city, war_start, inference_start))
        except FileNotFoundError:
            print(f"skipping {city}, no sentinel-1 data pulled for it yet")
    return pd.concat(tables, ignore_index=True)
