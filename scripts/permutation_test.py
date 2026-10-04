# checks whether the switching rule's apparent improvement (found by trying 6
# candidate thresholds on the same 19 cities we're reporting results on) is
# actually real, or just what you'd expect from luck plus trying several
# thresholds until one looks good.
#
# how: randomly shuffle which city has which damage rate (keeping the actual
# auc numbers fixed), then run the exact same "try 6 thresholds, keep the best"
# search on the shuffled, meaningless pairing. do that a few thousand times.
# if our real result isn't clearly bigger than most of these random, meaningless
# ones, that's a sign 19 cities just isn't enough data to trust the threshold.
#
# usage:
#   python scripts/permutation_test.py

from __future__ import annotations

import numpy as np
import pandas as pd

CANDIDATE_THRESHOLDS = [1, 2, 5, 8, 10, 15]
N_PERMUTATIONS = 5000


def best_switching_gain(pct_damaged: np.ndarray, baseline_auc: np.ndarray, model_auc: np.ndarray) -> float:
    always_baseline = baseline_auc.mean()
    best_gain = -np.inf
    for threshold in CANDIDATE_THRESHOLDS:
        use_model = pct_damaged > threshold
        switch_auc = np.where(use_model, model_auc, baseline_auc).mean()
        best_gain = max(best_gain, switch_auc - always_baseline)
    return best_gain


def main():
    df = pd.read_csv("data/cross_validation_results.csv")
    pct_damaged = df["pct_damaged"].to_numpy()
    baseline_auc = df["baseline_auc"].to_numpy()
    model_auc = df["gradient_boosting_auc"].to_numpy()

    observed_gain = best_switching_gain(pct_damaged, baseline_auc, model_auc)
    print(f"observed best switching gain (real data): {observed_gain*100:.3f} AUC points")

    rng = np.random.default_rng(0)
    permuted_gains = []
    for _ in range(N_PERMUTATIONS):
        shuffled_damage = rng.permutation(pct_damaged)
        permuted_gains.append(best_switching_gain(shuffled_damage, baseline_auc, model_auc))
    permuted_gains = np.array(permuted_gains)

    p_value = (permuted_gains >= observed_gain).mean()
    print(f"mean random gain: {permuted_gains.mean()*100:.3f}  "
          f"(random gains range from {permuted_gains.min()*100:.3f} to {permuted_gains.max()*100:.3f})")
    print(f"fraction of random shuffles that did AS WELL OR BETTER than the real result: {p_value*100:.1f}%")
    print()
    if p_value > 0.05:
        print("conclusion: the real result is NOT clearly better than random shuffling would produce.")
        print("19 cities is not enough data to trust this threshold as a real effect.")
    else:
        print("conclusion: the real result is unusually good compared to random shuffling.")


if __name__ == "__main__":
    main()
