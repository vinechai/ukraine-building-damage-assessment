# trains a few different classifiers on the features we built ourselves from the raw
# sentinel-1 data, and compares all of them against pwtt's own baseline rule. everything
# gets measured the exact same way, on cities the model never saw during training.
#
# note: max_change (pwtt's own score) is NOT used as a training feature for our models,
# only kept around to compute the baseline comparison. using it as a feature would be
# cheating, it would just be copying their answer instead of learning our own.
#
# usage:
#   python scripts/train_and_evaluate.py

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

# held out on purpose: never used for training, only for the final honest comparison.
# picked to cover a spread of damage rates, not just the two most famous cities.
TEST_CITIES = ["mariupol", "irpin", "sumy", "kramatorsk"]

FEATURE_COLS = [
    "pre_vv_mean", "pre_vv_std", "pre_vv_min", "pre_vv_max",
    "pre_vh_mean", "pre_vh_std", "pre_vh_min", "pre_vh_max",
    "post_vv_mean", "post_vv_std", "post_vh_mean", "post_vh_std",
    "wide_vv_mean", "wide_vv_std", "wide_vh_mean", "wide_vh_std",
    "diff_vv", "diff_vh", "z_vv", "z_vh", "combined_z",
    "diff_vv_wide", "diff_vh_wide", "z_vv_wide", "z_vh_wide", "combined_z_wide",
    "max_z_vv", "max_z_vh", "max_z_wide_vv", "max_z_wide_vh",
    "area",
]

MODELS = {
    "logistic regression": make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
    ),
    "random forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced", random_state=0, n_jobs=-1
    ),
    "gradient boosting": HistGradientBoostingClassifier(
        max_depth=6, class_weight="balanced", random_state=0
    ),
}


def evaluate(y_true, y_score, y_pred, label):
    auc = roc_auc_score(y_true, y_score)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    print(f"{label:<24}{auc*100:>8.2f}{f1*100:>8.2f}{precision*100:>10.2f}{recall*100:>10.2f}")
    return auc, f1


def main():
    df = pd.read_csv("data/features.csv")
    df = df.dropna(subset=FEATURE_COLS + ["damaged", "max_change"])

    train = df[~df["city"].isin(TEST_CITIES)]
    test = df[df["city"].isin(TEST_CITIES)]
    print(f"train: {len(train)} buildings from {train['city'].nunique()} cities")
    print(f"test:  {len(test)} buildings from {TEST_CITIES}")
    print()

    print(f"{'':24}{'AUC':>8}{'F1':>8}{'precision':>10}{'recall':>10}")
    baseline_pred = (test["max_change"] > 3.3).astype(int)
    evaluate(test["damaged"], test["max_change"], baseline_pred, "pwtt baseline (pooled)")

    fitted = {}
    for name, model in MODELS.items():
        model.fit(train[FEATURE_COLS], train["damaged"])
        fitted[name] = model
        score = model.predict_proba(test[FEATURE_COLS])[:, 1]
        pred = model.predict(test[FEATURE_COLS])
        evaluate(test["damaged"], score, pred, f"{name} (pooled)")

    print("\nper city:")
    for city in TEST_CITIES:
        sub = test[test["city"] == city]
        print(f"\n-- {city} ({len(sub)} buildings, {sub['damaged'].mean()*100:.2f}% damaged) --")
        sub_baseline_pred = (sub["max_change"] > 3.3).astype(int)
        evaluate(sub["damaged"], sub["max_change"], sub_baseline_pred, "  pwtt baseline")
        for name, model in fitted.items():
            score = model.predict_proba(sub[FEATURE_COLS])[:, 1]
            pred = model.predict(sub[FEATURE_COLS])
            evaluate(sub["damaged"], score, pred, f"  {name}")

    rf = fitted["random forest"]
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\nrandom forest feature importance (top 10):")
    print(importances.head(10))


if __name__ == "__main__":
    main()
