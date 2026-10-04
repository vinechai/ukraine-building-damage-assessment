# looks at real pwtt benchmark data for one or more ukrainian cities.
# also reproduces pwtt's own simple threshold rule so we can compare our own
# model against something later. this won't match their published per-city
# numbers exactly, they evaluate at 500m grid level with a threshold picked
# per city, we evaluate per building with one fixed threshold. see readme.
#
# usage:
#   python scripts/explore_benchmark.py --cities mariupol bucha irpin
#   python scripts/explore_benchmark.py --cities mariupol --plot --out mariupol.png

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from src.data.benchmark_labels import list_available_cities, load_city, pwtt_baseline_predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cities", nargs="+", default=None, help="city names, e.g. mariupol bucha. default: all")
    parser.add_argument("--plot", action="store_true", help="plot buildings colored by damage")
    parser.add_argument("--out", type=Path, default=None, help="save the plot instead of showing it")
    args = parser.parse_args()

    catalog = list_available_cities()
    if catalog.empty:
        print("no benchmark data found, run scripts/download_benchmark.py first")
        return

    cities = args.cities or sorted(catalog["city"])

    print(f"{'city':<16}{'n_buildings':>12}{'%damaged':>10}{'AUC':>8}{'F1':>8}{'precision':>10}{'recall':>10}")
    for city in cities:
        gdf = load_city(city)
        n = len(gdf)
        pct_damaged = 100 * gdf["damaged"].mean()

        y_true = gdf["damaged"]
        y_pred = pwtt_baseline_predictions(gdf)
        auc = roc_auc_score(y_true, gdf["max_change"])  # auc uses the raw score, not the thresholded 0/1
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        print(f"{city:<16}{n:>12}{pct_damaged:>9.2f}%{auc*100:>7.2f}{f1*100:>7.2f}{precision*100:>9.2f}{recall*100:>9.2f}")

        if len(cities) == 1:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            print(f"\nconfusion matrix (baseline rule, {city}):")
            print(f"                 predicted undamaged   predicted damaged")
            print(f"actual undamaged      {tn:>10}          {fp:>10}")
            print(f"actual damaged        {fn:>10}          {tp:>10}")

    if args.plot or args.out:
        import matplotlib.pyplot as plt

        gdfs = [load_city(c) for c in cities]
        fig, axes = plt.subplots(1, len(gdfs), figsize=(6 * len(gdfs), 6))
        if len(gdfs) == 1:
            axes = [axes]
        for ax, gdf, city in zip(axes, gdfs, cities):
            gdf[gdf["damaged"] == 0].plot(ax=ax, color="tab:green", markersize=1, label="undamaged")
            gdf[gdf["damaged"] == 1].plot(ax=ax, color="tab:red", markersize=1, label="damaged")
            ax.set_title(city)
            ax.set_axis_off()
            ax.legend(markerscale=8)
        plt.tight_layout()
        if args.out:
            plt.savefig(args.out, dpi=150)
            print(f"\nsaved plot to {args.out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()
