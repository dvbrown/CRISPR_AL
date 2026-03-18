"""Design B — feature optimisation script (cross-screen transfer).

Sweeps five feature configurations × Ridge × 5 seeds per direction to identify
the best feature set for cross-screen transfer before committing to the full
60-split Design B pipeline run.

Two directions evaluated:
  chen_to_sharon  Train on 2000 Chen genes, predict all Sharon genes (seeds 21001+)
  sharon_to_chen  Train on 2000 Sharon genes, predict all Chen genes (seeds 22001+)

Feature configs:
  coess_only      coessential_mean_r_top50, coessential_molm13_chronos
  expr_coess      above + molm13_log_tpm
  all_features    all 9 features
  coess_pathway   coess + pathway counts (no hallmark)
  expr_only       molm13_log_tpm

Primary ranking metric: P@50 SNR (mean / max(std, 0.02))

Sharon-only genes (absent from gene_features.parquet) are zero-imputed via
DataFrame.reindex(fill_value=0.0). The count of imputed genes is printed as a
diagnostic.

Usage:
    python optimize_design_b.py [--n-seeds N] [--output PATH]
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
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

SHARON_RAW = REPO / "data/bulk/sharon2019_venetoclax/BIOGRID-ORCS-SCREEN_1402-2.0.18.screen.tab.txt"

CHEN_SCREEN_ID = "chen2019_1393"
SHARON_SCREEN_ID = "sharon2019_1402"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
COESS = ["coessential_mean_r_top50", "coessential_molm13_chronos"]
PATHWAY_COUNTS = ["n_reactome_pathways", "n_go_bp_terms", "n_go_mf_terms", "n_kegg_pathways"]

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

FEATURE_SUBSETS = {
    "coess_only":    COESS,
    "expr_coess":    ["molm13_log_tpm"] + COESS,
    "all_features":  ALL_FEATURES,
    "coess_pathway": COESS + PATHWAY_COUNTS,
    "expr_only":     ["molm13_log_tpm"],
}

TRAIN_SIZE = 2000
SEED_START_CHEN_TO_SHARON = 21001   # aligned with splits.XFER_SEED_START
SEED_START_SHARON_TO_CHEN = 22001   # aligned with splits.XFER_SEED_START_REVERSE

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

# ---------------------------------------------------------------------------
# Cross-screen split runner
# ---------------------------------------------------------------------------

def run_cross_split(
    features_df: pd.DataFrame,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    seed: int,
    feature_cols: list[str],
    n_imputed_train: int,
    n_imputed_test: int,
) -> dict:
    """Single cross-screen split: sample 2000 train genes, test = all test genes.

    train_scores and test_scores must have columns: score_norm,
    is_hit_sensitizer, is_hit_resistor.

    features_df must be reindex'd to cover both screens (zero-imputed for
    Sharon-only genes).
    """
    rng = np.random.default_rng(seed)
    train_gene_pool = train_scores.index.tolist()
    sampled_train = set(rng.choice(train_gene_pool, size=min(TRAIN_SIZE, len(train_gene_pool)), replace=False))

    # Test genes: all test screen genes not in train
    test_genes = [g for g in test_scores.index if g not in sampled_train]

    X_tr = features_df.reindex(sorted(sampled_train), fill_value=0.0)[feature_cols].values
    X_te = features_df.reindex(test_genes, fill_value=0.0)[feature_cols].values
    y_tr = train_scores.loc[sorted(sampled_train), "score_norm"].values
    y_te = test_scores.loc[test_genes, "score_norm"].values
    hit_sens = test_scores.loc[test_genes, "is_hit_sensitizer"].values

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    return {
        "p50":    precision_at_k(y_pred, hit_sens, 50),
        "pearson": pearson_r(y_te, y_pred),
        "auroc":  auroc_sensitizer(y_pred, hit_sens),
        "seed":   seed,
        "n_test": len(test_genes),
    }

# ---------------------------------------------------------------------------
# Sweep over feature configs for one direction
# ---------------------------------------------------------------------------

def sweep_direction(
    features_df: pd.DataFrame,
    train_scores: pd.DataFrame,
    test_scores: pd.DataFrame,
    direction: str,
    seed_start: int,
    n_seeds: int,
    n_imputed_train: int,
    n_imputed_test: int,
) -> list[dict]:
    """Sweep all feature configs × n_seeds for one transfer direction."""
    rows = []
    total = len(FEATURE_SUBSETS) * n_seeds
    done = 0
    t0 = time.time()

    for feat_name, feat_cols in FEATURE_SUBSETS.items():
        seed_rows = []
        for i in range(n_seeds):
            seed = seed_start + i
            m = run_cross_split(
                features_df, train_scores, test_scores,
                seed, feat_cols, n_imputed_train, n_imputed_test,
            )
            seed_rows.append(m)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(
                f"\r  [{direction}] {done}/{total}  "
                f"elapsed {elapsed:.0f}s  eta {eta:.0f}s    ",
                end="", flush=True,
            )

        df = pd.DataFrame(seed_rows)
        p50_mean = df["p50"].mean()
        p50_std  = df["p50"].std()
        p50_snr  = p50_mean / max(p50_std, 0.02)
        rows.append({
            "feature_set":        feat_name,
            "direction":          direction,
            "p50_mean":           p50_mean,
            "p50_std":            p50_std,
            "p50_snr":            p50_snr,
            "pearson_mean":       df["pearson"].mean(),
            "auroc_sens_mean":    df["auroc"].mean(),
            "n_test_mean":        df["n_test"].mean(),
        })

    print()
    return rows

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5,
                        help="Number of random seeds per direction (default: 5)")
    parser.add_argument("--output", default=str(RESULTS / "optim_design_b_features.csv"),
                        help="Output CSV path")
    args = parser.parse_args()

    # --- Load Chen screen (pre-built parquet) ---
    print(f"Loading Chen screen from {ARTIFACTS}/screen_scores.parquet")
    chen_scores = pd.read_parquet(ARTIFACTS / "screen_scores.parquet")
    chen_scores = chen_scores.set_index("gene_symbol") if "gene_symbol" in chen_scores.columns else chen_scores
    required_cols = {"score_norm", "is_hit_sensitizer", "is_hit_resistor"}
    missing = required_cols - set(chen_scores.columns)
    if missing:
        print(f"ERROR: Chen screen parquet missing columns: {missing}", file=sys.stderr)
        sys.exit(1)
    print(f"  Chen: {len(chen_scores):,} genes")

    # --- Load Sharon screen (raw BioGRID file) ---
    print(f"Loading Sharon screen from {SHARON_RAW}")
    if not SHARON_RAW.exists():
        print(f"ERROR: Sharon raw file not found: {SHARON_RAW}", file=sys.stderr)
        sys.exit(1)

    from crispr_al.screen import load_sharon_screen_scores, zscore_normalize, assign_hit_labels
    sharon_raw = load_sharon_screen_scores(str(SHARON_RAW))
    sharon_raw = zscore_normalize(sharon_raw, score_col="lfc")
    sharon_raw = assign_hit_labels(sharon_raw)
    sharon_scores = sharon_raw.set_index("gene_symbol") if "gene_symbol" in sharon_raw.columns else sharon_raw
    print(f"  Sharon: {len(sharon_scores):,} genes")
    print(f"  Sharon sensitizers: {sharon_scores['is_hit_sensitizer'].sum()} "
          f"({sharon_scores['is_hit_sensitizer'].mean()*100:.1f}%)")

    # --- Load features ---
    print(f"\nLoading features from {ARTIFACTS}/gene_features.parquet")
    features_df = pd.read_parquet(ARTIFACTS / "gene_features.parquet")
    print(f"  Features: {len(features_df):,} genes, {features_df.shape[1]} columns")

    # --- Zero-imputation diagnostics ---
    chen_genes  = set(chen_scores.index)
    sharon_genes = set(sharon_scores.index)
    feat_genes  = set(features_df.index)

    n_chen_missing  = len(chen_genes  - feat_genes)
    n_sharon_missing = len(sharon_genes - feat_genes)
    print(f"\nFeature coverage:")
    print(f"  Chen genes missing from features:   {n_chen_missing:,}  (zero-imputed)")
    print(f"  Sharon genes missing from features: {n_sharon_missing:,}  (zero-imputed)")

    all_genes_needed = chen_genes | sharon_genes
    features_df = features_df.reindex(list(all_genes_needed), fill_value=0.0)

    # Confirm no NaN after imputation
    n_nan = features_df.isna().sum().sum()
    assert n_nan == 0, f"NaN values remain in features after reindex: {n_nan}"

    # --- Sweep Chen→Sharon ---
    print(f"\n=== Direction: chen_to_sharon (seeds {SEED_START_CHEN_TO_SHARON}+) ===")
    rows_c2s = sweep_direction(
        features_df,
        train_scores=chen_scores,
        test_scores=sharon_scores,
        direction="chen_to_sharon",
        seed_start=SEED_START_CHEN_TO_SHARON,
        n_seeds=args.n_seeds,
        n_imputed_train=n_chen_missing,
        n_imputed_test=n_sharon_missing,
    )

    # --- Sweep Sharon→Chen ---
    print(f"\n=== Direction: sharon_to_chen (seeds {SEED_START_SHARON_TO_CHEN}+) ===")
    rows_s2c = sweep_direction(
        features_df,
        train_scores=sharon_scores,
        test_scores=chen_scores,
        direction="sharon_to_chen",
        seed_start=SEED_START_SHARON_TO_CHEN,
        n_seeds=args.n_seeds,
        n_imputed_train=n_sharon_missing,
        n_imputed_test=n_chen_missing,
    )

    # --- Combine and save ---
    df = pd.DataFrame(rows_c2s + rows_s2c)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # --- Per-direction tables ---
    print_cols = ["feature_set", "p50_mean", "p50_std", "p50_snr", "pearson_mean", "auroc_sens_mean"]
    for direction in ("chen_to_sharon", "sharon_to_chen"):
        sub = df[df["direction"] == direction].sort_values("p50_snr", ascending=False)
        print(f"\n--- {direction} (ranked by P@50 SNR) ---")
        print(sub[print_cols].to_string(index=False, float_format="{:.4f}".format))

    # --- Combined ranking (average SNR across directions) ---
    combined = (
        df.groupby("feature_set")[["p50_mean", "p50_snr", "pearson_mean", "auroc_sens_mean"]]
        .mean()
        .rename(columns=lambda c: f"avg_{c}")
        .sort_values("avg_p50_snr", ascending=False)
        .reset_index()
    )
    print("\n--- Combined (avg across both directions, ranked by avg P@50 SNR) ---")
    print(combined.to_string(index=False, float_format="{:.4f}".format))

    best = combined.iloc[0]
    print(f"\n=== RECOMMENDATION ===")
    print(f"  Best feature set:     {best['feature_set']}")
    print(f"  Avg P@50:             {best['avg_p50_mean']:.4f}")
    print(f"  Avg P@50 SNR:         {best['avg_p50_snr']:.2f}")
    print(f"  Avg Pearson r:        {best['avg_pearson_mean']:.4f}")
    print(f"  Avg AUROC (sens):     {best['avg_auroc_sens_mean']:.4f}")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
