# the one chart that actually carries the project's finding: does our model do
# better relative to pwtt's baseline in cities with more real damage?
#
# usage:
#   python scripts/plot_damage_rate_vs_performance.py --out damage_rate_chart.png

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv("data/cross_validation_results.csv")
    df["auc_diff"] = df["gradient_boosting_auc"] - df["baseline_auc"]

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = df["auc_diff"].apply(lambda d: "tab:green" if d > 0 else "tab:red")
    ax.scatter(df["pct_damaged"], df["auc_diff"] * 100, c=colors, s=60, edgecolor="black", linewidth=0.5)

    for _, row in df.iterrows():
        ax.annotate(row["city"], (row["pct_damaged"], row["auc_diff"] * 100),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("% of buildings actually damaged in the city (log scale)")
    ax.set_ylabel("our model's AUC minus pwtt baseline's AUC\n(points above 0 = our model wins)")
    ax.set_title("model vs baseline, by city: low damage rates swing wildly, high rates lean slightly to the model")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
