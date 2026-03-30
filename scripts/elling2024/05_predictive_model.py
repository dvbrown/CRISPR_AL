"""Predictive model: classify in-vitro hits that validate in vivo (Elling 2024).

Label: is_validated_in_vivo = gene is a sensitiser hit in vivo (score_norm < -2.0).
Features: pathway features + in-vitro score magnitude + optional expression/co-essentiality.
Models: Logistic Regression and Random Forest (stratified 5-fold CV).
Baseline: predict by in-vitro score rank only (treats |score_norm_vitro| as the score).

Writes:
  - notebooks/crispr_star/results/predictive_model_results.parquet

Usage:
    source scripts/activate_env.sh
    python scripts/elling2024/05_predictive_model.py
    python scripts/elling2024/05_predictive_model.py --data-dir data/elling2024
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crispr_al.io import load_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/elling2024")
DEFAULT_RESULTS_DIR = Path("notebooks/crispr_star/results")
NORMZ_THRESHOLD = 2.0
N_FOLDS = 5
RANDOM_STATE = 42


def build_xy(scores_long: pd.DataFrame, features: pd.DataFrame) -> tuple:
    """Return (X, y, gene_symbols) for the predictive model.

    y = 1 if the gene is an in-vivo sensitiser hit (score_norm < -threshold).
    X = pathway/expression/co-essentiality features + abs(in-vitro score_norm).
    Only genes present in both in_vitro and in_vivo screens and in the feature
    matrix are included.
    """
    wide = scores_long.pivot_table(
        index="gene_symbol", columns="context", values="score_norm", aggfunc="mean"
    )
    wide.columns.name = None
    wide = wide.dropna(subset=["in_vitro", "in_vivo"])

    common_genes = wide.index.intersection(features.index)
    wide = wide.loc[common_genes]
    feat = features.loc[common_genes].fillna(0.0)

    X = feat.copy()
    X["abs_invitro_score"] = wide["in_vitro"].abs().values
    X["invitro_score"] = wide["in_vitro"].values

    y = (wide["in_vivo"] < -NORMZ_THRESHOLD).astype(int)
    log.info(
        "Dataset: %d genes, %d positives (%.1f%%)",
        len(y), y.sum(), 100 * y.mean(),
    )
    return X.values, y.values, X.columns.tolist(), wide["in_vitro"].values, list(common_genes)


def eval_fold(
    X_train, X_test, y_train, y_test, invitro_test,
    model_name: str, clf,
) -> dict:
    """Fit clf on train, evaluate on test. Returns metric dict."""
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)[:, 1]

    auroc = float(roc_auc_score(y_test, y_score)) if y_test.sum() > 0 else float("nan")
    auprc = float(average_precision_score(y_test, y_score)) if y_test.sum() > 0 else float("nan")
    # Naive baseline: predict by abs(invitro_score)
    baseline_score = np.abs(invitro_test)
    baseline_auroc = (
        float(roc_auc_score(y_test, baseline_score)) if y_test.sum() > 0 else float("nan")
    )
    baseline_auprc = (
        float(average_precision_score(y_test, baseline_score)) if y_test.sum() > 0 else float("nan")
    )

    return {
        "model": model_name,
        "auroc": auroc,
        "auprc": auprc,
        "baseline_auroc": baseline_auroc,
        "baseline_auprc": baseline_auprc,
        "n_test": int(len(y_test)),
        "n_positive_test": int(y_test.sum()),
    }


def main(data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_long = load_parquet(str(data_dir / "scores_long.parquet"))
    features = load_parquet(str(data_dir / "features_yumm17.parquet"))

    X, y, feature_names, invitro_scores, genes = build_xy(scores_long, features)

    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=1.0)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        log.info("Fold %d/%d", fold_idx + 1, N_FOLDS)
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        invitro_te = invitro_scores[test_idx]

        for name, clf in models.items():
            import copy
            result = eval_fold(X_tr, X_te, y_tr, y_te, invitro_te, name, copy.deepcopy(clf))
            result["fold"] = fold_idx
            rows.append(result)

    results_df = pd.DataFrame(rows)

    summary = (
        results_df.groupby("model")[["auroc", "auprc", "baseline_auroc", "baseline_auprc"]]
        .agg(["mean", "std"])
    )
    log.info("CV summary:\n%s", summary.to_string())

    save_parquet(results_df, str(results_dir / "predictive_model_results.parquet"))
    log.info("Saved predictive_model_results.parquet")
    log.info("Next step: python scripts/elling2024/06_plot_results.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Elling2024 data directory")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Results output directory")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.results_dir))
