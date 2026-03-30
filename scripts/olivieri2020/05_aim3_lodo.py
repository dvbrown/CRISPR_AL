"""Aim 3: Leave-one-drug-out (LODO) evaluation for Olivieri 2020.

For each library (TKOv2, TKOv3): train on all other screens in the same
library, predict on the held-out screen. Training labels are Z-normalised
per screen before stacking, then predictions are back-transformed to the
test screen's NormZ scale for evaluation.

Expected output: results/olivieri2020/aim3_lodo_results.parquet
  60 rows (30 screens × 2 models)
Expected performance: RF strong (TKOv2 Pearson ≈ 0.34, AUROC ≈ 0.84;
  TKOv3 Pearson ≈ 0.26, AUROC ≈ 0.83). Ridge near-chance.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from crispr_al.io import load_parquet, save_parquet
from crispr_al.models import predict, scale_features, train_rf, train_ridge
from crispr_al.splits import generate_lodo_splits

from utils import FEATURE_COLS, RIDGE_ALPHAS, compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _build_screen_cache(normz: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Pre-compute per-screen X/y (Z-normalised labels) for all screens.

    Caching avoids rebuilding the same arrays from scratch for every LODO split.
    Without per-screen normalisation, Ridge learns near-zero coefficients because
    stacked labels average to ~0 across genes.
    """
    cache = {}
    for sc in normz.columns:
        y_s = normz[sc].dropna()
        common = features.index.intersection(y_s.index)
        y_vals = y_s.loc[common].values.astype(float)
        std = y_vals.std()
        y_z = (y_vals - y_vals.mean()) / std if std > 0 else y_vals - y_vals.mean()
        cache[sc] = (
            features.loc[common, FEATURE_COLS].values.astype(float),
            y_z,
        )
    return cache


def main(data_dir: str, results_dir: str) -> None:
    data = Path(data_dir)
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)

    normz = load_parquet(str(data / "normz_matrix.parquet"))
    screen_meta = load_parquet(str(data / "screen_metadata.parquet"))
    features = load_parquet(str(data / "features_6pathway.parquet"))

    screen_cache = _build_screen_cache(normz, features)

    rows = []
    for library in ("TKOv2", "TKOv3"):
        splits = generate_lodo_splits(normz, screen_meta, library)
        for split in splits:
            test_screen = split["test_screen"]
            train_screens = split["train_screens"]

            y_te_series = normz[test_screen].dropna()
            common_te = features.index.intersection(y_te_series.index)
            X_te = features.loc[common_te, FEATURE_COLS].values.astype(float)
            y_te = y_te_series.loc[common_te].values.astype(float)

            X_parts = [screen_cache[sc][0] for sc in train_screens]
            y_parts = [screen_cache[sc][1] for sc in train_screens]
            X_tr = np.vstack(X_parts)
            y_tr = np.concatenate(y_parts)

            # Predictions back-transformed to test screen's NormZ scale
            te_mean, te_std = y_te.mean(), y_te.std()

            X_tr_s, X_te_s = scale_features(X_tr, X_te)
            ridge = train_ridge(X_tr_s, y_tr, alphas=RIDGE_ALPHAS)
            pred_ridge = predict(ridge, X_te_s) * te_std + te_mean
            rows.append({
                **compute_metrics(y_te, pred_ridge),
                "test_screen": test_screen, "library": library, "model": "Ridge",
            })

            rf = train_rf(X_tr, y_tr, seed=42, min_samples_leaf=5)
            pred_rf = predict(rf, X_te) * te_std + te_mean
            rows.append({
                **compute_metrics(y_te, pred_rf),
                "test_screen": test_screen, "library": library, "model": "RF",
            })

            logger.info("LODO %s/%s done", library, test_screen)

    results = pd.DataFrame(rows)
    save_parquet(results, str(out / "aim3_lodo_results.parquet"))
    for lib in ("TKOv2", "TKOv3"):
        for mdl in ("RF", "Ridge"):
            sub = results[(results["library"] == lib) & (results["model"] == mdl)]
            logger.info(
                "%s %s: Pearson=%.3f, AUROC=%.3f",
                lib, mdl, sub["pearson"].mean(), sub["auroc"].mean(),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", default="data/olivieri2020",
        help="Directory containing normz_matrix, screen_metadata, and features files (default: data/olivieri2020)",
    )
    parser.add_argument(
        "--results-dir", default="results/olivieri2020",
        help="Output directory for results (default: results/olivieri2020)",
    )
    args = parser.parse_args()
    main(args.data_dir, args.results_dir)
