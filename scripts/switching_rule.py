# turns the "we win more often in higher-damage cities" finding into an actual rule:
# above some damage-rate threshold, trust our model; below it, trust pwtt's baseline.
#
# honest limitation, worth stating plainly: the threshold below is picked by looking
# at the same 19 cities we're reporting results on, not validated on a separate set.
# this is a description of a real pattern in our results, not a newly proven claim.
# it also assumes you already have a rough idea of a city's damage rate before
# picking a method, which in a real deployment you might not have upfront (that's
# often the very thing you're trying to find out). a reasonable proxy in practice
# would be something like known conflict duration/intensity for that area.
#
# usage:
#   python scripts/switching_rule.py

from __future__ import annotations

import pandas as pd

CANDIDATE_THRESHOLDS = [1, 2, 5, 8, 10, 15]


def main():
    df = pd.read_csv("data/cross_validation_results.csv")

    print(f"{'threshold':>10}{'switching AUC':>16}{'switching F1':>15}")
    always_baseline_auc = df["baseline_auc"].mean()
    always_model_auc = df["gradient_boosting_auc"].mean()
    always_baseline_f1 = df["baseline_f1"].mean()
    always_model_f1 = df["gradient_boosting_f1"].mean()
    print(f"{'always baseline':>10}{always_baseline_auc*100:>16.2f}{always_baseline_f1*100:>15.2f}")
    print(f"{'always model':>10}{always_model_auc*100:>16.2f}{always_model_f1*100:>15.2f}")
    print()

    best_threshold, best_auc = None, -1
    for threshold in CANDIDATE_THRESHOLDS:
        use_model = df["pct_damaged"] > threshold
        switch_auc = df["gradient_boosting_auc"].where(use_model, df["baseline_auc"])
        switch_f1 = df["gradient_boosting_f1"].where(use_model, df["baseline_f1"])
        print(f"{threshold:>9}%{switch_auc.mean()*100:>16.2f}{switch_f1.mean()*100:>15.2f}"
              f"   ({use_model.sum()} of 19 cities use the model)")
        if switch_auc.mean() > best_auc:
            best_auc, best_threshold = switch_auc.mean(), threshold

    print(f"\nbest threshold found: {best_threshold}% damage rate")
    use_model = df["pct_damaged"] > best_threshold
    df["method_used"] = use_model.map({True: "model", False: "baseline"})
    print(df[["city", "pct_damaged", "method_used", "baseline_auc", "gradient_boosting_auc"]]
          .sort_values("pct_damaged", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
