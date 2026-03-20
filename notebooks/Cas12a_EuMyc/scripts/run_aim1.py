"""Run Aim 1 within-screen holdout experiments for the Cas12a/EuMyc workstream.

For a given arm and selection strategy, trains Ridge or RF models on a
randomly selected or pathway-stratified subset of ~2,000 genes and evaluates
prediction of the remaining genome-wide gene scores.

Usage:
  source scripts/activate_env.sh
  python notebooks/Cas12a_EuMyc/scripts/run_aim1.py \\
      --arm menuetto_nutlin --strategy random --model ridge --n-repeats 5

Output files per repeat/model:
  results/aim1/{arm}/{repeat:03d}_{strategy}_{model}.json
  results/aim1/{arm}/{repeat:03d}_{strategy}_{model}_predictions.csv
"""
import argparse
import datetime
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Repository root is 3 levels up from this script
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from crispr_al.io import save_metrics_json, get_code_commit
from crispr_al.models import scale_features, train_ridge, train_rf, predict
from crispr_al.metrics import (
    compute_regression_metrics,
    compute_ranking_metrics,
    compute_classification_metrics,
    build_metrics_record,
)
from crispr_al.splits import (
    split_random,
    split_reactome_stratified,
    split_hallmark_stratified,
    split_apoptosis_p53_seeded,
    split_bcl2_seeded,
    split_reactome_apoptosis_oversampled,
    compute_split_hash,
)

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_ROOT = ROOT / "data" / "bulk"
PROC_DIR  = DATA_ROOT / "menuetto_scherzo_2025" / "processed"

REACTOME_PATH  = DATA_ROOT / "pathway_annotations" / "NCBI2Reactome_PE_Pathway.txt.gz"
HALLMARKS_PATH = DATA_ROOT / "pathway_annotations" / "h.all.v2024.1.Hs.symbols.gmt.gz"

# Default seed namespace for Aim 1 EuMyc (avoids collision with venetoclax seeds)
DEFAULT_SEED_START = 31001

# Arm → (parquet file, lfc column)
ARM_CONFIG = {
    "menuetto_nutlin":  ("menuetto_gene_scores.parquet",          "lfc_nutlin"),
    "menuetto_s63845":  ("menuetto_gene_scores.parquet",          "lfc_s63845"),
    "scherzo_nutlin":   ("scherzo_gene_scores.parquet",           "lfc_nutlin"),
    "scherzo_s63845":   ("scherzo_gene_scores.parquet",           "lfc_s63845"),
    "imdf_t3":          ("imdf_gene_scores_by_timepoint.parquet", "lfc_t3"),
}

# Strategy names that require Reactome membership
REACTOME_STRATEGIES = {"reactome_stratified", "reactome_apoptosis_oversampled"}
# Strategy names that require Hallmark membership
HALLMARK_STRATEGIES = {"hallmark_stratified", "apoptosis_p53_seeded", "bcl2_seeded"}


# ── Data loading helpers ──────────────────────────────────────────────────────

def _parse_gmt(path: Path) -> dict:
    """Parse a gzipped GMT file. Returns {gene_set_name: [gene_symbols]}."""
    gene_sets = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                gene_sets[parts[0]] = list(parts[2:])
    return gene_sets


def _parse_reactome(path: Path) -> dict:
    """Parse NCBI2Reactome_PE_Pathway.txt.gz.

    Format: EntrezID  ReactomeID  URL  PathwayName  Evidence  Species
    Returns {pathway_id: [gene_symbols]} using the gene symbol from URL column
    where possible, falling back to Entrez ID. Filters for Homo sapiens.
    """
    membership: dict = {}
    with gzip.open(path, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            species = parts[5].strip()
            if species != "Homo sapiens":
                continue
            # parts[0] = Entrez/symbol, parts[1] = pathway stable ID
            gene_sym = parts[0].strip()
            pid = parts[1].strip()
            membership.setdefault(pid, []).append(gene_sym)
    return membership


def load_reactome_membership(ortholog_df: pd.DataFrame) -> dict:
    """Load Reactome pathway membership, translated to mouse gene symbols.

    Human gene symbols in Reactome are mapped to mouse orthologs. Only mouse
    genes that appear in the ortholog map are retained.
    """
    human_to_mouse = dict(
        zip(ortholog_df["human_symbol"], ortholog_df["mouse_symbol"])
    )
    raw = _parse_reactome(REACTOME_PATH)
    translated: dict = {}
    for pid, genes in raw.items():
        mouse_members = [human_to_mouse[g] for g in genes if g in human_to_mouse]
        if mouse_members:
            translated[pid] = mouse_members
    return translated


def load_hallmark_membership(ortholog_df: pd.DataFrame) -> dict:
    """Load MSigDB Hallmark gene sets, translated to mouse gene symbols."""
    human_to_mouse = dict(
        zip(ortholog_df["human_symbol"], ortholog_df["mouse_symbol"])
    )
    raw = _parse_gmt(HALLMARKS_PATH)
    translated: dict = {}
    for name, genes in raw.items():
        mouse_members = [human_to_mouse[g] for g in genes if g in human_to_mouse]
        if mouse_members:
            translated[name] = mouse_members
    return translated


# ── Split function dispatch ───────────────────────────────────────────────────

def call_split_function(
    strategy: str,
    gene_list: list,
    n_train: int,
    seed: int,
    params: dict,
    reactome_membership: dict,
    hallmark_membership: dict,
) -> tuple:
    """Dispatch to the appropriate split function by strategy name."""
    if strategy == "random":
        return split_random(gene_list, n_train, seed)

    if strategy == "reactome_stratified":
        return split_reactome_stratified(
            gene_list, reactome_membership, n_train, seed,
            min_pathway_size=params.get("min_pathway_size", 10),
            max_pathway_size=params.get("max_pathway_size", 500),
        )

    if strategy == "hallmark_stratified":
        return split_hallmark_stratified(
            gene_list, hallmark_membership, n_train, seed,
            n_per_hallmark=params.get("n_per_hallmark", 5),
        )

    if strategy == "apoptosis_p53_seeded":
        return split_apoptosis_p53_seeded(
            gene_list, hallmark_membership, n_train, seed,
            seed_hallmarks=params.get("seed_hallmarks"),
            max_seed_fraction=params.get("max_seed_fraction", 0.20),
        )

    if strategy == "bcl2_seeded":
        return split_bcl2_seeded(
            gene_list, hallmark_membership, n_train, seed,
            seed_hallmarks=params.get("seed_hallmarks"),
            max_seed_fraction=params.get("max_seed_fraction", 0.20),
        )

    if strategy == "reactome_apoptosis_oversampled":
        return split_reactome_apoptosis_oversampled(
            gene_list, reactome_membership, n_train, seed,
            n_per_apoptosis=params.get("n_per_apoptosis", 5),
            n_per_other=params.get("n_per_other", 1),
            min_pathway_size=params.get("min_pathway_size", 10),
            max_pathway_size=params.get("max_pathway_size", 500),
        )

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True,
                        choices=list(ARM_CONFIG.keys()),
                        help="Screen arm to evaluate")
    parser.add_argument("--strategy", required=True,
                        choices=["random", "reactome_stratified", "hallmark_stratified",
                                 "apoptosis_p53_seeded", "bcl2_seeded",
                                 "reactome_apoptosis_oversampled"],
                        help="Gene selection strategy")
    parser.add_argument("--model", default="both",
                        choices=["ridge", "rf", "both"],
                        help="Model(s) to train (default: both)")
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Number of repeats (default: from splits.yaml)")
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START,
                        help=f"Starting RNG seed (default: {DEFAULT_SEED_START})")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "notebooks" / "Cas12a_EuMyc" / "results" / "aim1",
                        help="Output directory (default: notebooks/Cas12a_EuMyc/results/aim1/)")
    parser.add_argument("--splits-yaml", type=Path,
                        default=ROOT / "notebooks" / "Cas12a_EuMyc" / "splits.yaml",
                        help="Path to splits.yaml registry")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned runs without executing")
    args = parser.parse_args()

    # Load splits config
    with open(args.splits_yaml) as f:
        splits_cfg = yaml.safe_load(f)

    aim_cfg      = splits_cfg["aim1"]
    strategy_cfg = aim_cfg["strategies"][args.strategy]
    n_train      = aim_cfg["n_train"]

    # Arm constraint check
    allowed_arms = strategy_cfg.get("arms")
    if allowed_arms and args.arm not in allowed_arms:
        print(f"  ERROR: strategy '{args.strategy}' is only valid for arms "
              f"{allowed_arms}. Got '{args.arm}'.", file=sys.stderr)
        sys.exit(1)

    n_repeats = args.n_repeats or strategy_cfg.get("n_repeats", aim_cfg["n_repeats"])
    params    = strategy_cfg.get("params", {})
    models    = ["ridge", "rf"] if args.model == "both" else [args.model]

    if args.dry_run:
        print(f"Dry run: arm={args.arm}  strategy={args.strategy}  "
              f"models={models}  n_repeats={n_repeats}  n_train={n_train}")
        for i in range(n_repeats):
            seed = args.seed_start + i
            for model in models:
                stem = f"{i+1:03d}_{args.strategy}_{model}"
                out_dir = args.output_dir / args.arm
                print(f"  repeat {i+1:3d}  seed={seed}  → {out_dir / stem}.json")
        return

    # ── Load data ──────────────────────────────────────────────────────────────
    parquet_file, lfc_col = ARM_CONFIG[args.arm]
    print(f"Loading gene scores: {parquet_file}  column: {lfc_col}")
    scores_df = pd.read_parquet(PROC_DIR / parquet_file)
    scores_df = scores_df.dropna(subset=[lfc_col])

    print("Loading feature matrix ...")
    features_df = pd.read_parquet(PROC_DIR / "features_mouse_genes.parquet")

    # Align genes: must be in both score and feature matrices
    common_genes = scores_df.index.intersection(features_df.index)
    scores_df    = scores_df.loc[common_genes]
    features_df  = features_df.loc[common_genes]
    gene_list    = list(common_genes)
    print(f"  {len(gene_list)} genes after alignment")

    # Hit labels: bottom 5% = sensitizer, top 5% = resistor
    lfc = scores_df[lfc_col].values
    p5  = np.percentile(lfc, 5)
    p95 = np.percentile(lfc, 95)
    hit_sens_all = (lfc <= p5).astype(bool)
    hit_res_all  = (lfc >= p95).astype(bool)
    print(f"  sensitizer hits: {hit_sens_all.sum()} (≤ {p5:.3f})")
    print(f"  resistor  hits: {hit_res_all.sum()} (≥ {p95:.3f})")

    feature_cols = features_df.columns.tolist()
    X_all = features_df[feature_cols].values.astype(np.float64)

    # ── Load pathway membership (lazy, only if needed) ────────────────────────
    reactome_membership: dict = {}
    hallmark_membership: dict = {}

    if args.strategy in REACTOME_STRATEGIES or args.strategy in HALLMARK_STRATEGIES:
        print("Loading ortholog map ...")
        ortholog_df = pd.read_parquet(PROC_DIR / "mouse_human_orthologs.parquet")

    if args.strategy in REACTOME_STRATEGIES:
        print("Parsing Reactome pathways ...")
        reactome_membership = load_reactome_membership(ortholog_df)
        print(f"  {len(reactome_membership)} pathways loaded")

    if args.strategy in HALLMARK_STRATEGIES:
        print("Parsing Hallmark gene sets ...")
        hallmark_membership = load_hallmark_membership(ortholog_df)
        print(f"  {len(hallmark_membership)} Hallmark sets loaded")

    # ── Output directory ───────────────────────────────────────────────────────
    out_dir = args.output_dir / args.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    code_commit = get_code_commit()
    screen_id   = f"eumyc_{args.arm}"

    # ── Run repeats ────────────────────────────────────────────────────────────
    for i in range(n_repeats):
        seed = args.seed_start + i

        train_genes, holdout_genes = call_split_function(
            args.strategy, gene_list, n_train, seed,
            params, reactome_membership, hallmark_membership,
        )
        assert len(set(train_genes) & set(holdout_genes)) == 0, (
            f"Train/holdout overlap in repeat {i+1}!"
        )

        train_idx   = [gene_list.index(g) for g in train_genes]
        holdout_idx = [gene_list.index(g) for g in holdout_genes]

        X_train = X_all[train_idx]
        X_test  = X_all[holdout_idx]
        y_train = lfc[train_idx]
        y_test  = lfc[holdout_idx]
        hs_test = hit_sens_all[holdout_idx]
        hr_test = hit_res_all[holdout_idx]

        X_train_s, X_test_s = scale_features(X_train, X_test)

        split_hash = compute_split_hash(
            f"aim1_eumyc_{args.strategy}", screen_id, seed, train_genes
        )
        split_dict = {
            "split_id":       f"aim1_{args.arm}_{args.strategy}_r{i+1:03d}",
            "generator_id":   f"aim1_eumyc_{args.strategy}",
            "family":         args.strategy,
            "aim":            "aim1_eumyc",
            "metrics_profile": "aim1_transfer",
            "seed":           seed,
            "repeat_index":   i + 1,
            "train_screen_id": screen_id,
            "test_screen_id":  screen_id,
            "split_hash":     split_hash,
        }

        data_counts = {
            "train_row_count":           len(train_genes),
            "test_row_count":            len(holdout_genes),
            "n_unique_train_genes":      len(set(train_genes)),
            "n_unique_test_genes":       len(set(holdout_genes)),
            "n_overlap_genes_train_test": 0,
        }
        leakage_checks = {
            "disjoint_gene_label_rows":      True,
            "normalization_fit_on_train_only": True,
            "split_hash_logged":             True,
        }

        for model_name in models:
            stem = f"{i+1:03d}_{args.strategy}_{model_name}"
            out_json  = out_dir / f"{stem}.json"
            out_preds = out_dir / f"{stem}_predictions.csv"

            if out_json.exists():
                print(f"  skip (exists) repeat {i+1} / {model_name}")
                continue

            if model_name == "ridge":
                model_obj = train_ridge(X_train_s, y_train)
            else:
                model_obj = train_rf(X_train_s, y_train, seed=seed)

            y_pred = predict(model_obj, X_test_s)

            reg  = compute_regression_metrics(y_test, y_pred)
            rank = compute_ranking_metrics(y_pred, hs_test, hr_test)
            clf  = compute_classification_metrics(y_pred, hs_test, hr_test)

            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            record = build_metrics_record(
                split=split_dict,
                data_counts=data_counts,
                leakage_checks=leakage_checks,
                regression=reg,
                ranking=rank,
                classification=clf,
                run_id=f"{split_dict['split_id']}_{model_name}",
                timestamp_utc=timestamp,
                code_commit=code_commit,
            )
            save_metrics_json(record, str(out_json))

            preds_df = pd.DataFrame({
                "gene_symbol":      holdout_genes,
                "y_test":           y_test,
                "y_pred":           y_pred,
                "is_hit_sensitizer": hs_test,
                "is_hit_resistor":   hr_test,
            })
            preds_df.to_csv(out_preds, index=False)

            print(
                f"  repeat {i+1:3d} / {model_name:<5}  "
                f"Pearson={reg['pearson']:.3f}  "
                f"AUROC_sens={clf['labels'][0]['auroc']:.3f}  "
                f"P@50={rank['k_metrics'][0]['precision_at_k']:.3f}"
            )

    print("Done.")


if __name__ == "__main__":
    main()
