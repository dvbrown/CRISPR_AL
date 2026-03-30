"""Aim 2: Cross-library transfer evaluation for Olivieri 2020.

Trains on one library version of a drug and predicts on the other.
Uses all available genes in each screen (no subsampling).

Cross-library pairs:
  Cisplatin TKOv2 ↔ Cisplatin TKOv3 rep A/B (4 directions)
  Camptothecin TKOv2 ↔ Camptothecin TKOv3 (2 directions)

Expected output: results/olivieri2020/aim2_cross_library_results.parquet
  12 rows (6 pairs × 2 models)
Expected performance: RF substantially better than Ridge
  (Cisplatin AUROC ≈ 0.76, Camptothecin AUROC ≈ 0.74).
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from crispr_al.io import load_parquet, save_parquet
from crispr_al.models import predict, scale_features, train_rf, train_ridge

from utils import FEATURE_COLS, RIDGE_ALPHAS, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CROSS_LIB_PAIRS = [
    ("Cisplatin_TKOv2",       "Cisplatin2_repA_TKOv3", "Cisplatin"),
    ("Cisplatin_TKOv2",       "Cisplatin2_repB_TKOv3", "Cisplatin"),
    ("Cisplatin2_repA_TKOv3", "Cisplatin_TKOv2",       "Cisplatin"),
    ("Cisplatin2_repB_TKOv3", "Cisplatin_TKOv2",       "Cisplatin"),
    ("Camptothecin_TKOv2",    "Camptothecin2_TKOv3",   "Camptothecin"),
    ("Camptothecin2_TKOv3",   "Camptothecin_TKOv2",    "Camptothecin"),
]


def main(data_dir: str, results_dir: str) -> None:
    data = Path(data_dir)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    normz = load_parquet(str(data / "normz_matrix.parquet"))
    features = load_parquet(str(data / "features_6pathway.parquet"))

    rows = []
    for train_screen, test_screen, drug in CROSS_LIB_PAIRS:
        y_tr_series = normz[train_screen].dropna()
        y_te_series = normz[test_screen].dropna()
        common_tr = features.index.intersection(y_tr_series.index)
        common_te = features.index.intersection(y_te_series.index)

        X_tr = features.loc[common_tr, FEATURE_COLS].values.astype(float)
        y_tr = y_tr_series.loc[common_tr].values.astype(float)
        X_te = features.loc[common_te, FEATURE_COLS].values.astype(float)
        y_te = y_te_series.loc[common_te].values.astype(float)

        X_tr_s, X_te_s = scale_features(X_tr, X_te)
        ridge = train_ridge(X_tr_s, y_tr, alphas=RIDGE_ALPHAS)
        rows.append({
            **compute_metrics(y_te, predict(ridge, X_te_s)),
            "train_screen": train_screen, "test_screen": test_screen,
            "drug": drug, "model": "Ridge",
        })

        rf = train_rf(X_tr, y_tr, seed=42, min_samples_leaf=5)
        rows.append({
            **compute_metrics(y_te, predict(rf, X_te)),
            "train_screen": train_screen, "test_screen": test_screen,
            "drug": drug, "model": "RF",
        })

        logger.info("%s → %s done", train_screen, test_screen)

    results = pd.DataFrame(rows)
    save_parquet(results, str(out / "aim2_cross_library_results.parquet"))
    logger.info(
        "Aim 2 complete: %d rows. Median AUROC RF=%.3f, Ridge=%.3f",
        len(results),
        results.loc[results["model"] == "RF", "auroc"].median(),
        results.loc[results["model"] == "Ridge", "auroc"].median(),
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
