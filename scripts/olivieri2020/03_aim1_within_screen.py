"""Aim 1: Within-screen holdout evaluation across all 30 Olivieri 2020 screens.

For each screen independently: 25 repeats of 80/20 random gene holdout,
training Ridge and Random Forest regressors on the 6 pathway features.

Expected output: results/olivieri2020/aim1_within_screen_results.parquet
  1500 rows (30 screens × 25 repeats × 2 models)
Expected performance: near-chance (Pearson ≈ 0.03–0.06, AUROC ≈ 0.59).
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from crispr_al.io import load_parquet, save_parquet
from crispr_al.models import predict, scale_features, train_rf, train_ridge

from utils import FEATURE_COLS, RIDGE_ALPHAS, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

N_REPEATS = 25
TEST_SIZE = 0.2


def main(data_dir: str, results_dir: str) -> None:
    data = Path(data_dir)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    normz = load_parquet(str(data / "normz_matrix.parquet"))
    features = load_parquet(str(data / "features_6pathway.parquet"))

    rows = []
    for screen_col in normz.columns:
        y_series = normz[screen_col].dropna()
        common = features.index.intersection(y_series.index)
        X_all = features.loc[common, FEATURE_COLS].values.astype(float)
        y_all = y_series.loc[common].values.astype(float)

        for rep in range(N_REPEATS):
            seed = rep * 137 + abs(hash(screen_col)) % 9973
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_all, y_all, test_size=TEST_SIZE, random_state=seed
            )

            X_tr_s, X_te_s = scale_features(X_tr, X_te)
            ridge = train_ridge(X_tr_s, y_tr, alphas=RIDGE_ALPHAS)
            rows.append({
                **compute_metrics(y_te, predict(ridge, X_te_s)),
                "screen": screen_col, "model": "Ridge", "repeat": rep,
            })

            rf = train_rf(X_tr, y_tr, seed=seed, min_samples_leaf=5)
            rows.append({
                **compute_metrics(y_te, predict(rf, X_te)),
                "screen": screen_col, "model": "RF", "repeat": rep,
            })

        logger.info("Done: %s", screen_col)

    results = pd.DataFrame(rows)
    save_parquet(results, str(out / "aim1_within_screen_results.parquet"))
    logger.info(
        "Aim 1 complete: %d rows. Median Pearson=%.3f, AUROC=%.3f",
        len(results),
        results["pearson"].median(),
        results["auroc"].median(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data/olivieri2020",
        help="Directory containing normz_matrix and features parquet files (default: data/olivieri2020)",
    )
    parser.add_argument(
        "--results-dir", default="results/olivieri2020",
        help="Output directory for results (default: results/olivieri2020)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
