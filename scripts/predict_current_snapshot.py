# applies the trained model to a fresh, current sentinel-1 window and draws a map of
# what it predicts. this is NOT a validated result: there's no unosat-style manual
# assessment of the present day to check it against. it's shown as an illustration of
# the tool actually being pointed at something current, not a proven finding.
#
# usage:
#   python scripts/predict_current_snapshot.py --city rubizhne --out docs/maps/rubizhne_current.html

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import folium
import geopandas as gpd
import pandas as pd

from src.data.benchmark_labels import load_city
from src.data.build_features import build_snapshot_features
from train_and_evaluate import FEATURE_COLS, MODELS

WAR_START = "2022-02-22"


def _swatch(color: str) -> str:
    return (f'<span style="background:{color};width:11px;height:11px;display:inline-block;'
            f'border-radius:50%;border:1px solid rgba(0,0,0,0.3);forced-color-adjust:none;"></span>')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    city = args.city.lower()

    warnings.filterwarnings("ignore", category=UserWarning)

    snapshot_path = Path(f"data/sentinel1/{city}_current_timeseries.csv")
    if not snapshot_path.exists():
        print(f"no current snapshot pulled yet for {city}, run scripts/pull_current_snapshot.py first")
        return

    raw_snapshot_dates = pd.read_csv(snapshot_path, parse_dates=["date"])["date"]
    date_range = f"{raw_snapshot_dates.min():%b %Y} to {raw_snapshot_dates.max():%b %Y}"

    # match THIS city's actual training pass count, not a generic default. checked
    # directly and it varies a lot per city (kremenchuk's real training window had
    # exactly 1 pass, rubizhne's had 2, mariupol's had 3), because it depends on
    # whatever satellite coverage happened to exist for that city's specific
    # inference_start date back in 2022. using the wrong count re-introduces the same
    # kind of mismatch the pass-count fix was meant to solve.
    # same fix, same reasoning, for the wide window: match this city's own real
    # training wide-window pass count instead of using every pass we happened to pull.
    city_features = pd.read_csv("data/features.csv")
    city_rows = city_features[city_features["city"] == city]
    narrow_passes = int(city_rows["n_post"].mode().iloc[0]) if not city_rows.empty else 2
    wide_passes = int(city_rows["n_wide"].mode().iloc[0]) if not city_rows.empty else None

    snapshot = build_snapshot_features(
        city, WAR_START, snapshot_path, narrow_passes=narrow_passes, wide_passes=wide_passes
    ).dropna(subset=FEATURE_COLS)

    train = pd.read_csv("data/features.csv").dropna(subset=FEATURE_COLS + ["damaged", "max_change"])
    model = MODELS["gradient boosting"].__class__(**MODELS["gradient boosting"].get_params())
    model.fit(train[FEATURE_COLS], train["damaged"])

    snapshot["predicted_damaged"] = model.predict(snapshot[FEATURE_COLS])
    snapshot["predicted_prob"] = model.predict_proba(snapshot[FEATURE_COLS])[:, 1]

    # a pwtt-style rule for comparison: "was there a single date where this building's
    # radar reading jumped further than 3.3 pre-war standard deviations." this is NOT
    # pwtt's actual formula (we never reimplemented their specific multi-scale
    # weighting), it's our own peak-change statistic using their published threshold
    # convention, labeled as such, not presented as their real algorithm.
    snapshot["pwtt_style_pred"] = (snapshot[["max_z_wide_vv", "max_z_wide_vh"]].max(axis=1) > 3.3).astype(int)

    geo = load_city(city)[["FID", "geometry"]]
    plot_df = gpd.GeoDataFrame(snapshot.merge(geo, left_on="building_id", right_on="FID"), geometry="geometry", crs="EPSG:4326")
    center = [plot_df.geometry.centroid.y.mean(), plot_df.geometry.centroid.x.mean()]

    # this city's own leave-one-out result from 2022 (data/cross_validation_results.csv),
    # the closest thing to an honest confidence statement we can make: "here's how
    # reliable this exact method was, the one time we could actually check it here"
    cv = pd.read_csv("data/cross_validation_results.csv")
    cv_row = cv[cv["city"] == city].iloc[0] if (cv["city"] == city).any() else None

    historical_pct = train[train["city"] == city]["damaged"].mean() * 100 if (train["city"] == city).any() else None
    current_pct = plot_df["predicted_damaged"].mean() * 100
    pwtt_style_pct = plot_df["pwtt_style_pred"].mean() * 100

    m = folium.Map(location=center, zoom_start=13, tiles=None)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                      attr="Esri, Maxar, Earthstar Geographics", name="satellite", control=False).add_to(m)

    for layer_name, pred_col in [("our model", "predicted_damaged"), ("pwtt-style rule", "pwtt_style_pred")]:
        group = folium.FeatureGroup(name=layer_name, show=(layer_name == "our model"))
        for value, color in [(0, "#cccccc"), (1, "#d62728")]:
            subset = plot_df[plot_df[pred_col] == value]
            if subset.empty:
                continue
            folium.GeoJson(
                subset[["geometry"]].__geo_interface__,
                style_function=lambda _f, color=color: {"fillColor": color, "color": color, "weight": 1.3, "fillOpacity": 0.2},
            ).add_to(group)
        group.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    warning_html = (
        '<div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);z-index:1000;'
        'background:#fff3cd;border:2px solid #d62728;padding:10px 16px;border-radius:4px;'
        'font-family:sans-serif;font-size:13px;max-width:640px;text-align:center;">'
        f'<b>UNVALIDATED</b>, no human has confirmed real damage for this time period. shown as an '
        f'illustration of the tool in use, not a proven result. the satellite photo underneath is '
        f'Esri\'s basemap, whose exact capture date is unknown to us and not guaranteed to be recent.</div>'
    )
    m.get_root().html.add_child(folium.Element(warning_html))

    confidence_line = ""
    if cv_row is not None:
        confidence_line = (
            f'<hr style="margin:6px 0;">'
            f'<b>context from the one time we could check this method here (2022):</b><br>'
            f'AUC {cv_row["gradient_boosting_auc"]*100:.1f}, '
            f'precision {cv_row["gradient_boosting_precision"]*100:.1f}%, '
            f'recall {cv_row["gradient_boosting_recall"]*100:.1f}%<br>'
            f'<i>treat this snapshot with at least that much uncertainty, and likely more, '
            f'since this compares against a period the model was never trained to compare against.</i>'
        )

    info_html = (
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;'
        f'padding:10px 12px;border-radius:4px;font-family:sans-serif;font-size:12px;max-width:320px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
        f'<b>{args.city}, {date_range} (unvalidated)</b><br>'
        + (f'{historical_pct:.1f}% actually damaged in 2022 (real, validated)<br>' if historical_pct is not None else "")
        + f'our model flags {current_pct:.1f}%, pwtt-style rule flags {pwtt_style_pct:.1f}% '
          f'({date_range}, no ground truth for either)'
        f'{confidence_line}'
        f'<hr style="margin:6px 0;">'
        f'<div>{_swatch("#cccccc")} predicted undamaged</div>'
        f'<div>{_swatch("#d62728")} predicted damaged</div>'
        f'</div>'
    )
    m.get_root().html.add_child(folium.Element(info_html))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(args.out))
    print(f"saved {args.out} ({len(plot_df)} buildings, narrow/wide windows matched to "
          f"{narrow_passes}/{wide_passes} passes like this city's own 2022 training data). "
          f"our model: {current_pct:.1f}%, pwtt-style rule: {pwtt_style_pct:.1f}%, "
          f"historical (2022): {historical_pct:.1f}%")


if __name__ == "__main__":
    main()
