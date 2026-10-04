# builds one interactive html map per city: buildings colored by whether each method
# got them right, not by what it predicted. lets you actually see where pwtt's
# baseline and our model make mistakes, and how those mistakes differ, on a real
# satellite basemap. a toggle switches between "pwtt baseline errors" and
# "our model errors" against the same ground truth underneath.
#
# static html output, works on github pages, no server needed to view it.
#
# usage:
#   python scripts/build_interactive_map.py --city rubizhne --out docs/maps/rubizhne.html

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
from train_and_evaluate import FEATURE_COLS, MODELS

# free satellite basemap, no api key needed
ESRI_WORLD_IMAGERY = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics"

# damaged = red, undamaged = gray, everywhere in this project, no exceptions. an
# earlier version used green for "correct damaged" which read backwards (red usually
# means damage/danger) and was inconsistent with the current-snapshot maps anyway.
# the two categories that involve real damage (correct damaged, false negative) are
# the two most visually loud colors (red, purple) since they're the important ones.
# an earlier attempt used two similar muted tones here and they were nearly
# indistinguishable from each other and from gray at low opacity. false positive
# gets a plain, distinct yellow so it's still identifiable, just not as loud.
CATEGORY_COLORS = {
    "correct undamaged": "#cccccc",
    "false positive (false alarm)": "#f2c200",
    "false negative (missed real damage)": "#7b1fa2",
    "correct damaged": "#d62728",
}


def _categorize(truth: pd.Series, pred: pd.Series) -> pd.Series:
    return pd.Series(
        [
            "correct damaged" if t == 1 and p == 1 else
            "correct undamaged" if t == 0 and p == 0 else
            "false positive (false alarm)" if t == 0 and p == 1 else
            "false negative (missed real damage)"
            for t, p in zip(truth, pred)
        ],
        index=truth.index,
    )


def _swatch(color: str) -> str:
    # forced-color-adjust:none stops browsers in dark/high-contrast mode from
    # stripping the background color off small elements like this. without it,
    # windows dark theme (or similar) can silently turn every swatch gray.
    return (f'<span style="background:{color};width:11px;height:11px;display:inline-block;'
            f'border-radius:50%;border:1px solid rgba(0,0,0,0.3);forced-color-adjust:none;"></span>')


def build_error_map(city: str, out_path: Path):
    warnings.filterwarnings("ignore", category=UserWarning)  # geographic-crs centroid warning, harmless here

    df = pd.read_csv("data/features.csv").dropna(subset=FEATURE_COLS + ["damaged", "max_change"])
    train = df[df["city"] != city]
    test = df[df["city"] == city].copy()

    model = MODELS["gradient boosting"].__class__(**MODELS["gradient boosting"].get_params())
    model.fit(train[FEATURE_COLS], train["damaged"])
    test["model_pred"] = model.predict(test[FEATURE_COLS])
    test["baseline_pred"] = (test["max_change"] > 3.3).astype(int)

    test["baseline_category"] = _categorize(test["damaged"], test["baseline_pred"])
    test["model_category"] = _categorize(test["damaged"], test["model_pred"])

    geo = load_city(city)[["FID", "geometry"]]
    plot_df = gpd.GeoDataFrame(test.merge(geo, left_on="building_id", right_on="FID"), geometry="geometry", crs="EPSG:4326")
    center = [plot_df.geometry.centroid.y.mean(), plot_df.geometry.centroid.x.mean()]

    m = folium.Map(location=center, zoom_start=13, tiles=None)
    folium.TileLayer(tiles=ESRI_WORLD_IMAGERY, attr=ESRI_ATTRIBUTION, name="satellite", control=False).add_to(m)

    # actual building outlines, not dots. this makes each city's file noticeably
    # bigger (a building polygon has ~8 points on average vs 1 for a dot), but shows
    # the real building shapes on top of the satellite photo instead of an abstract
    # scatter of circles
    for layer_name, category_col in [("pwtt baseline errors", "baseline_category"), ("our model errors", "model_category")]:
        group = folium.FeatureGroup(name=layer_name, show=(layer_name == "pwtt baseline errors"))
        for category, color in CATEGORY_COLORS.items():
            subset = plot_df[plot_df[category_col] == category]
            if subset.empty:
                continue
            folium.GeoJson(
                subset[["geometry"]].__geo_interface__,
                style_function=lambda _f, color=color: {
                    "fillColor": color, "color": color, "weight": 1.3, "fillOpacity": 0.2,
                },
            ).add_to(group)
        group.add_to(m)

    baseline_pct = test["baseline_pred"].mean() * 100
    model_pct = test["model_pred"].mean() * 100
    actual_pct = plot_df["damaged"].mean() * 100
    legend_rows = "".join(f'<div>{_swatch(color)} {cat}</div>' for cat, color in CATEGORY_COLORS.items())
    info_html = (
        f'<div style="position:fixed;bottom:20px;left:20px;z-index:1000;background:white;'
        f'padding:10px 12px;border-radius:4px;font-family:sans-serif;font-size:12px;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.3);">'
        f'<b>{city}, 2022</b><br>'
        f'{actual_pct:.1f}% actually damaged (real, 2022 unosat assessment)<br>'
        f'pwtt baseline flagged {baseline_pct:.1f}% &middot; our model flagged {model_pct:.1f}%'
        f'<hr style="margin:6px 0;">{legend_rows}</div>'
    )
    m.get_root().html.add_child(folium.Element(info_html))

    folium.LayerControl(collapsed=False).add_to(m)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(out_path))
    print(f"saved {out_path} ({len(plot_df)} buildings)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    build_error_map(args.city, args.out)


if __name__ == "__main__":
    main()
