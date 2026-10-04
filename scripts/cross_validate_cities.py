# tests the model properly: for each of the 19 cities, train on the other 18 and
# evaluate on that one city, never seen during that round of training. repeated 19
# times so every city gets a turn as the "new, unseen place." much sturdier than
# picking a handful of test cities by hand, and still doesn't need any new data,
# everything runs on data/features.csv we already built.
#
# usage:
#   python scripts/cross_validate_cities.py

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

from train_and_evaluate import FEATURE_COLS, MODELS


def score(y_true, y_score, y_pred):
    return dict(
        auc=roc_auc_score(y_true, y_score),
        f1=f1_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
    )


RESULTS_PATH = Path("data/cross_validation_results.csv")


def main():
    df = pd.read_csv("data/features.csv")
    df = df.dropna(subset=FEATURE_COLS + ["damaged", "max_change"])
    cities = sorted(df["city"].unique())

    # save after every city instead of only at the end, so an interrupted run
    # (closed laptop, closed session) doesn't lose everything already computed
    done_cities = set()
    if RESULTS_PATH.exists():
        done_cities = set(pd.read_csv(RESULTS_PATH)["city"])
        print(f"resuming: {len(done_cities)} cities already done: {sorted(done_cities)}")

    for i, city in enumerate(cities):
        if city in done_cities:
            continue
        train = df[df["city"] != city]
        test = df[df["city"] == city]

        baseline_pred = (test["max_change"] > 3.3).astype(int)
        result = {"city": city, "n": len(test), "pct_damaged": test["damaged"].mean() * 100}
        result.update({f"baseline_{k}": v for k, v in score(test["damaged"], test["max_change"], baseline_pred).items()})

        for name, model_template in MODELS.items():
            model = clone(model_template)
            model.fit(train[FEATURE_COLS], train["damaged"])
            model_score = model.predict_proba(test[FEATURE_COLS])[:, 1]
            model_pred = model.predict(test[FEATURE_COLS])
            key = name.replace(" ", "_")
            result.update({f"{key}_{k}": v for k, v in score(test["damaged"], model_score, model_pred).items()})

        pd.DataFrame([result]).to_csv(RESULTS_PATH, mode="a", header=not RESULTS_PATH.exists(), index=False)
        print(f"[{i+1}/{len(cities)}] {city} done "
              f"(baseline AUC {result['baseline_auc']*100:.1f}, "
              f"gradient_boosting AUC {result['gradient_boosting_auc']*100:.1f})")

    results = pd.read_csv(RESULTS_PATH)

    print("\n=== summary across all 19 cities (each held out once) ===")
    for method in ["baseline"] + [m.replace(" ", "_") for m in MODELS]:
        auc_col, f1_col = f"{method}_auc", f"{method}_f1"
        print(f"{method:<24} mean AUC={results[auc_col].mean()*100:.2f}  mean F1={results[f1_col].mean()*100:.2f}")

    print("\nhow many of the 19 cities did each method win on AUC:")
    for method in [m.replace(" ", "_") for m in MODELS]:
        wins = (results[f"{method}_auc"] > results["baseline_auc"]).sum()
        print(f"  {method}: won {wins}/19")


if __name__ == "__main__":
    main()
