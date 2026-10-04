# draws one city as 3 side by side maps: real damage (ground truth), what pwtt's
# baseline rule predicted, and what our trained model predicted. makes the "when does
# our model actually help" finding visible directly instead of needing a table.
#
# retrains the model leaving this city out entirely, same as the proper cross
# validation, so the map shows genuine unseen-city predictions, not the model
# grading its own training data.
#
# usage:
#   python scripts/plot_city_comparison.py --city mariupol --out mariupol_comparison.png
#   python scripts/plot_city_comparison.py --city chernihiv --out chernihiv_comparison.png

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.benchmark_labels import load_city
from train_and_evaluate import FEATURE_COLS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--city", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    city = args.city.lower()

    df = pd.read_csv("data/features.csv").dropna(subset=FEATURE_COLS + ["damaged", "max_change"])
    train = df[df["city"] != city]
    test = df[df["city"] == city]

    model = HistGradientBoostingClassifier(max_depth=6, class_weight="balanced", random_state=0)
    model.fit(train[FEATURE_COLS], train["damaged"])
    test = test.copy()
    test["model_pred"] = model.predict(test[FEATURE_COLS])
    test["baseline_pred"] = (test["max_change"] > 3.3).astype(int)

    geo = load_city(city)[["FID", "geometry"]]
    plot_df = test.merge(geo, left_on="building_id", right_on="FID")
    import geopandas as gpd
    plot_df = gpd.GeoDataFrame(plot_df, geometry="geometry", crs="EPSG:4326")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    panels = [("damaged", "ground truth"), ("baseline_pred", "pwtt baseline"), ("model_pred", "our model")]
    for ax, (col, title) in zip(axes, panels):
        plot_df[plot_df[col] == 0].plot(ax=ax, color="tab:green", markersize=1)
        plot_df[plot_df[col] == 1].plot(ax=ax, color="tab:red", markersize=1)
        ax.set_title(f"{args.city}: {title}")
        ax.set_axis_off()

    pct_damaged = test["damaged"].mean() * 100
    fig.suptitle(f"{args.city} ({pct_damaged:.1f}% actually damaged), green=undamaged, red=damaged")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
