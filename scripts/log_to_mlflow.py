# logs the already-computed cross validation results into mlflow, so the comparison
# between pwtt's baseline, our 3 models, and the switching rule is browsable properly
# instead of living only in a csv. doesn't retrain anything, just records results we
# already have.
#
# usage:
#   python scripts/log_to_mlflow.py
#   mlflow ui   (then open the printed local url to browse it)

from __future__ import annotations

import mlflow
import pandas as pd

SWITCHING_THRESHOLD = 10  # % damage rate, see scripts/switching_rule.py for why


def log_method(df: pd.DataFrame, method: str, auc_col: str, f1_col: str, precision_col: str, recall_col: str):
    with mlflow.start_run(run_name=method, nested=False):
        mlflow.log_param("method", method)
        mlflow.log_metric("mean_auc", df[auc_col].mean())
        mlflow.log_metric("mean_f1", df[f1_col].mean())
        mlflow.log_metric("mean_precision", df[precision_col].mean())
        mlflow.log_metric("mean_recall", df[recall_col].mean())

        for _, row in df.iterrows():
            with mlflow.start_run(run_name=f"{method} - {row['city']}", nested=True):
                mlflow.log_param("method", method)
                mlflow.log_param("city", row["city"])
                mlflow.log_param("n_buildings", int(row["n"]))
                mlflow.log_param("pct_damaged", round(row["pct_damaged"], 3))
                mlflow.log_metric("auc", row[auc_col])
                mlflow.log_metric("f1", row[f1_col])
                mlflow.log_metric("precision", row[precision_col])
                mlflow.log_metric("recall", row[recall_col])


def main():
    df = pd.read_csv("data/cross_validation_results.csv")
    mlflow.set_experiment("ukraine-building-damage")

    log_method(df, "pwtt_baseline", "baseline_auc", "baseline_f1", "baseline_precision", "baseline_recall")
    log_method(df, "logistic_regression", "logistic_regression_auc", "logistic_regression_f1",
               "logistic_regression_precision", "logistic_regression_recall")
    log_method(df, "random_forest", "random_forest_auc", "random_forest_f1",
               "random_forest_precision", "random_forest_recall")
    log_method(df, "gradient_boosting", "gradient_boosting_auc", "gradient_boosting_f1",
               "gradient_boosting_precision", "gradient_boosting_recall")

    # switching rule: model above the threshold, baseline below it. built by picking
    # whichever column applies per row, not by retraining or re-predicting anything.
    use_model = df["pct_damaged"] > SWITCHING_THRESHOLD
    switch = df.copy()
    for metric in ["auc", "f1", "precision", "recall"]:
        switch[f"switch_{metric}"] = df[f"gradient_boosting_{metric}"].where(use_model, df[f"baseline_{metric}"])
    log_method(switch, f"switching_rule_{SWITCHING_THRESHOLD}pct",
               "switch_auc", "switch_f1", "switch_precision", "switch_recall")

    print("logged. run `mlflow ui` and open the local url it prints to browse the comparison.")


if __name__ == "__main__":
    main()
