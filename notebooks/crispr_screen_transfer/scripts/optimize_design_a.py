"""Design A — parameter optimisation script.

Iterates over feature subsets, train sizes, and model hyperparameters using a
small number of seeds for fast feedback. Reports a ranked leaderboard by mean
Precision@50 (primary) and Pearson r (secondary) on the held-out gene set.

Three sequential phases:
  Phase 1  Feature selection — all feature subsets, fixed train_size=2000,
           baseline model set.
  Phase 2  Train-size sweep — best feature subset from Phase 1, all train
           sizes, best model from Phase 1.
  Phase 3  Model hyperparameter search — best features + train size, full
           model grid.

Usage:
    python optimize_design_a.py [--n-seeds N] [--phase 1|2|3|all]
                                [--features-override KEY]
                                [--train-size-override N]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[3]
ARTIFACTS = REPO / "notebooks/crispr_screen_transfer/artifacts"
RESULTS = REPO / "notebooks/crispr_screen_transfer/results"
RESULTS.mkdir(exist_ok=True)

SEED_START = 11001  # aligned with splits.SEED_START

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
ALL_FEATURES = [
    "molm13_log_tpm",
    "coessential_mean_r_top50",
    "coessential_molm13_chronos",
    "n_reactome_pathways",
    "n_go_bp_terms",
    "n_go_mf_terms",
    "in_hallmark_apoptosis",
    "in_hallmark_oxidative_phosphorylation",
    "n_kegg_pathways",
]

COESS = ["coessential_mean_r_top50", "coessential_molm13_chronos"]
PATHWAY = [
    "n_reactome_pathways", "n_go_bp_terms", "n_go_mf_terms",
    "in_hallmark_apoptosis", "in_hallmark_oxidative_phosphorylation",
    "n_kegg_pathways",
]

FEATURE_SUBSETS = {
    "all_9":          ALL_FEATURES,
    "expr_only":      ["molm13_log_tpm"],
    "coess_only":     COESS,
    "pathway_only":   PATHWAY,
    "expr_coess":     ["molm13_log_tpm"] + COESS,
    "expr_top_coess": ["molm13_log_tpm", "coessential_mean_r_top50"],
    "expr_chronos":   ["molm13_log_tpm", "coessential_molm13_chronos"],
    "no_expr":        [f for f in ALL_FEATURES if f != "molm13_log_tpm"],
    "no_coess":       ["molm13_log_tpm"] + PATHWAY,
    "no_go":          [f for f in ALL_FEATURES if "n_go" not in f],
    "no_hallmark":    [f for f in ALL_FEATURES if "hallmark" not in f],
    "no_kegg":        [f for f in ALL_FEATURES if f != "n_kegg_pathways"],
    "no_reactome":    [f for f in ALL_FEATURES if f != "n_reactome_pathways"],
}

TRAIN_SIZES = [500, 1000, 2000, 3000, 5000]

# ---------------------------------------------------------------------------
# Model factory: returns (label, fitted_model) after fit
# Models that use StandardScaler internally are flagged with scale=True
# ---------------------------------------------------------------------------
def _make_models(seed: int) -> list[tuple[str, object, bool]]:
    """Return list of (name, unfitted_model, needs_scaling)."""
    return [
        ("ridge_default",
         RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5), True),
        ("ridge_wide",
         RidgeCV(alphas=np.logspace(-3, 4, 15), cv=5), True),
        ("elasticnet",
         ElasticNetCV(
             l1_ratio=[0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95],
             alphas=np.logspace(-4, 2, 12),
             cv=5, max_iter=10_000,
         ), True),
        ("rf_sqrt",
         RandomForestRegressor(
             n_estimators=100, max_features="sqrt",
             random_state=seed, n_jobs=-1,
         ), False),
        ("rf_log2",
         RandomForestRegressor(
             n_estimators=100, max_features="log2",
             random_state=seed, n_jobs=-1,
         ), False),
        ("rf_30pct",
         RandomForestRegressor(
             n_estimators=100, max_features=0.3,
             random_state=seed, n_jobs=-1,
         ), False),
        ("rf_all_feats",
         RandomForestRegressor(
             n_estimators=100, max_features=1.0,
             random_state=seed, n_jobs=-1,
         ), False),
        ("rf_leaf5",
         RandomForestRegressor(
             n_estimators=100, max_features="sqrt", min_samples_leaf=5,
             random_state=seed, n_jobs=-1,
         ), False),
        ("rf_depth10",
         RandomForestRegressor(
             n_estimators=100, max_features="sqrt", max_depth=10,
             random_state=seed, n_jobs=-1,
         ), False),
        ("lgbm_default",
         lgb.LGBMRegressor(
             n_estimators=200, num_leaves=31,
             random_state=seed, n_jobs=-1, verbosity=-1,
         ), False),
        ("lgbm_shallow",
         lgb.LGBMRegressor(
             n_estimators=300, num_leaves=15, max_depth=4,
             min_child_samples=20,
             random_state=seed, n_jobs=-1, verbosity=-1,
         ), False),
        ("lgbm_deep",
         lgb.LGBMRegressor(
             n_estimators=500, num_leaves=63, max_depth=8,
             learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             random_state=seed, n_jobs=-1, verbosity=-1,
         ), False),
        ("lgbm_l1reg",
         lgb.LGBMRegressor(
             n_estimators=200, num_leaves=31, reg_alpha=1.0,
             random_state=seed, n_jobs=-1, verbosity=-1,
         ), False),
    ]

# Phase 1 uses a representative subset to keep runtime short
PHASE1_MODELS = {
    "ridge_default", "ridge_wide", "elasticnet",
    "rf_sqrt", "rf_all_feats",
    "lgbm_default", "lgbm_shallow",
}

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_at_k(y_pred: np.ndarray, hit: np.ndarray, k: int = 50) -> float:
    """Precision@K for sensitizers (lowest predicted scores)."""
    order = np.argsort(y_pred)
    n = min(k, len(y_pred))
    return float(hit[order[:n]].sum() / n)


def pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def auroc_sensitizer(y_pred: np.ndarray, hit: np.ndarray) -> float:
    if hit.sum() == 0 or hit.sum() == len(hit):
        return 0.5
    return float(roc_auc_score(hit.astype(int), -y_pred))


def evaluate(y_true, y_pred, hit_sens, hit_res) -> dict:
    return {
        "p50":   precision_at_k(y_pred, hit_sens, 50),
        "p100":  precision_at_k(y_pred, hit_sens, 100),
        "pearson": pearson_r(y_true, y_pred),
        "auroc": auroc_sensitizer(y_pred, hit_sens),
    }

# ---------------------------------------------------------------------------
# Core: single split evaluation
# ---------------------------------------------------------------------------

def run_split(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    seed: int,
    train_size: int,
    feature_cols: list[str],
    model_name: str,
    model,
    needs_scaling: bool,
) -> dict:
    rng = np.random.default_rng(seed)
    all_genes = features.index.tolist()
    train_genes = set(rng.choice(all_genes, size=train_size, replace=False))
    test_genes = [g for g in all_genes if g not in train_genes]

    X_tr = features.loc[sorted(train_genes), feature_cols].values
    X_te = features.loc[test_genes, feature_cols].values
    y_tr = scores.loc[sorted(train_genes), "score_norm"].values
    y_te = scores.loc[test_genes, "score_norm"].values
    hit_sens = scores.loc[test_genes, "is_hit_sensitizer"].values
    hit_res  = scores.loc[test_genes, "is_hit_resistor"].values

    if needs_scaling:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    m = evaluate(y_te, y_pred, hit_sens, hit_res)
    m["seed"] = seed
    m["model"] = model_name
    return m

# ---------------------------------------------------------------------------
# Sweep helper
# ---------------------------------------------------------------------------

def sweep(
    features: pd.DataFrame,
    scores: pd.DataFrame,
    configs: list[dict],
    n_seeds: int,
    tag: str,
) -> pd.DataFrame:
    """Run all configs × seeds, return results DataFrame."""
    rows = []
    total = len(configs) * n_seeds
    done = 0
    t0 = time.time()

    for cfg in configs:
        seed_rows = []
        for i in range(n_seeds):
            seed = SEED_START + i
            all_models = _make_models(seed)
            model_map = {name: (m, sc) for name, m, sc in all_models}
            mname = cfg["model"]
            if mname not in model_map:
                continue
            model, needs_scaling = model_map[mname]

            m = run_split(
                features, scores, seed,
                cfg["train_size"], cfg["features"],
                mname, model, needs_scaling,
            )
            seed_rows.append(m)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(
                f"\r  [{tag}] {done}/{total}  "
                f"elapsed {elapsed:.0f}s  eta {eta:.0f}s    ",
                end="", flush=True,
            )

        if not seed_rows:
            continue
        df = pd.DataFrame(seed_rows)
        p50_mean = df["p50"].mean()
        p50_std  = df["p50"].std()
        # Signal-to-noise: penalise high-variance results (floor std at 0.02)
        p50_snr  = p50_mean / max(p50_std, 0.02)
        rows.append({
            "features":     cfg["feat_name"],
            "n_features":   len(cfg["features"]),
            "train_size":   cfg["train_size"],
            "model":        cfg["model"],
            "p50_mean":     p50_mean,
            "p50_std":      p50_std,
            "p50_snr":      p50_snr,
            "p100_mean":    df["p100"].mean(),
            "pearson_mean": df["pearson"].mean(),
            "auroc_mean":   df["auroc"].mean(),
        })

    print()
    return pd.DataFrame(rows).sort_values("p50_snr", ascending=False)

# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase1(features, scores, n_seeds, feat_override=None):
    """Feature selection: all subsets × Phase1 models at train_size=2000."""
    print("\n=== PHASE 1: Feature selection (train_size=2000) ===")
    subsets = FEATURE_SUBSETS if feat_override is None else {feat_override: FEATURE_SUBSETS[feat_override]}
    configs = [
        {"feat_name": fname, "features": fcols, "train_size": 2000, "model": mname}
        for fname, fcols in subsets.items()
        for mname in PHASE1_MODELS
    ]
    return sweep(features, scores, configs, n_seeds, "phase1")


def phase2(features, scores, n_seeds, best_feat, best_model, size_override=None, model_override=None):
    """Train-size sweep: best features × all train sizes × best model."""
    model = model_override or best_model
    print(f"\n=== PHASE 2: Train-size sweep (features={best_feat}, model={model}) ===")
    sizes = [size_override] if size_override else TRAIN_SIZES
    configs = [
        {"feat_name": best_feat, "features": FEATURE_SUBSETS[best_feat], "train_size": ts, "model": model}
        for ts in sizes
    ]
    return sweep(features, scores, configs, n_seeds, "phase2")


def phase_grid(features, scores, n_seeds, feat_keys, model_names, train_sizes=None):
    """Joint sweep: selected feature sets × models × train sizes."""
    sizes = train_sizes or TRAIN_SIZES
    print(f"\n=== GRID SWEEP: {feat_keys} × {model_names} × {sizes} ===")
    configs = [
        {"feat_name": fk, "features": FEATURE_SUBSETS[fk], "train_size": ts, "model": mn}
        for fk in feat_keys
        for mn in model_names
        for ts in sizes
    ]
    return sweep(features, scores, configs, n_seeds, "grid")


def phase3(features, scores, n_seeds, best_feat, best_size):
    """Full model hyperparameter grid: best features + train size × all models."""
    print(f"\n=== PHASE 3: Model hyperparam search (features={best_feat}, train_size={best_size}) ===")
    all_model_names = [name for name, _, _ in _make_models(SEED_START)]
    configs = [
        {"feat_name": best_feat, "features": FEATURE_SUBSETS[best_feat], "train_size": best_size, "model": mname}
        for mname in all_model_names
    ]
    return sweep(features, scores, configs, n_seeds, "phase3")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame, title: str, top_n: int = 20):
    print(f"\n--- {title} (top {top_n}, ranked by P@50 SNR) ---")
    cols = ["features", "model", "train_size", "p50_mean", "p50_std", "p50_snr", "p100_mean", "pearson_mean", "auroc_mean"]
    cols = [c for c in cols if c in df.columns]
    print(df[cols].head(top_n).to_string(index=False, float_format="{:.4f}".format))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--phase", choices=["1", "2", "3", "all", "grid"], default="all")
    parser.add_argument("--features-override", default=None,
                        help="Fix feature subset key (skips Phase 1 search)")
    parser.add_argument("--train-size-override", type=int, default=None,
                        help="Fix train size (skips Phase 2 search)")
    parser.add_argument("--model-override", default=None,
                        help="Force a specific model in Phase 2")
    args = parser.parse_args()

    print(f"Loading features and scores from {ARTIFACTS}")
    features = pd.read_parquet(ARTIFACTS / "gene_features.parquet")
    scores   = pd.read_parquet(ARTIFACTS / "screen_scores.parquet")
    # Align
    genes = features.index.intersection(scores.index)
    features = features.loc[genes]
    scores   = scores.loc[genes]
    print(f"  {len(genes):,} genes, {features.shape[1]} features")

    run_phases = set(args.phase.split() if args.phase not in ("all", "grid") else [])
    if args.phase == "all":
        run_phases = {"1", "2", "3"}
    elif args.phase == "grid":
        run_phases = {"grid"}

    results = {}

    # Grid sweep (alternative to phased approach)
    if "grid" in run_phases:
        top_feats  = ["coess_only", "expr_coess", "expr_chronos", "no_expr", "no_reactome"]
        top_models = ["ridge_default", "ridge_wide", "elasticnet",
                      "rf_sqrt", "rf_all_feats", "lgbm_default", "lgbm_shallow"]
        dfg = phase_grid(features, scores, args.n_seeds, top_feats, top_models)
        print_table(dfg, "Grid sweep")
        dfg.to_csv(RESULTS / "optim_grid.csv", index=False)
        best = dfg.iloc[0]
        print(f"\n=== GRID RECOMMENDATION ===")
        print(f"  Feature set:  {best['features']}  ({best.get('n_features', '?')} features)")
        print(f"  Train size:   {best['train_size']}")
        print(f"  Model:        {best['model']}")
        print(f"  P@50:         {best['p50_mean']:.4f} ± {best.get('p50_std', 0):.4f}")
        print(f"  P@50 SNR:     {best['p50_snr']:.2f}")
        print(f"  Pearson r:    {best['pearson_mean']:.4f}")
        print(f"  AUROC:        {best['auroc_mean']:.4f}")
        print(f"\nResults saved to {RESULTS}")
        return

    # Phase 1
    if "1" in run_phases:
        df1 = phase1(features, scores, args.n_seeds, args.features_override)
        print_table(df1, "Phase 1: Feature selection")
        df1.to_csv(RESULTS / "optim_phase1_features.csv", index=False)
        best_feat  = df1.iloc[0]["features"]
        best_model = df1.iloc[0]["model"]
        print(f"\n  → Best feature set: {best_feat}")
        print(f"  → Best model:       {best_model}")
        results["phase1"] = df1
    else:
        best_feat  = args.features_override or "all_9"
        best_model = "lgbm_default"

    # Phase 2
    if "2" in run_phases:
        df2 = phase2(features, scores, args.n_seeds, best_feat, best_model,
                     args.train_size_override, args.model_override)
        print_table(df2, "Phase 2: Train-size sweep")
        df2.to_csv(RESULTS / "optim_phase2_train_size.csv", index=False)
        best_size = int(df2.iloc[0]["train_size"])
        print(f"\n  → Best train size: {best_size}")
        results["phase2"] = df2
    else:
        best_size = args.train_size_override or 2000

    # Phase 3
    if "3" in run_phases:
        df3 = phase3(features, scores, args.n_seeds, best_feat, best_size)
        print_table(df3, "Phase 3: Model hyperparameter search")
        df3.to_csv(RESULTS / "optim_phase3_models.csv", index=False)
        results["phase3"] = df3

    # Final summary
    print("\n=== FINAL RECOMMENDATION ===")
    if "3" in run_phases:
        best = results["phase3"].iloc[0]
    elif "2" in run_phases:
        best = results["phase2"].iloc[0]
    else:
        best = results["phase1"].iloc[0]

    print(f"  Feature set:  {best['features']}  ({best.get('n_features', '?')} features)")
    print(f"  Train size:   {best['train_size']}")
    print(f"  Model:        {best['model']}")
    print(f"  P@50:         {best['p50_mean']:.4f} ± {best.get('p50_std', 0):.4f}")
    print(f"  Pearson r:    {best['pearson_mean']:.4f}")
    print(f"  AUROC:        {best['auroc_mean']:.4f}")
    print(f"\nResults saved to {RESULTS}")


if __name__ == "__main__":
    main()
