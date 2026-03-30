"""Compute in-vitro / in-vivo agreement metrics for Elling 2024 (CRISPR-StAR).

Reads data/elling2024/scores_long.parquet and computes:
  - Global agreement (Pearson r, Spearman rho)
  - Concordance-at-the-top (CAT@N) curve
  - Hit overlap Jaccard index (NormZ threshold: ±2.0)
  - Per-gene discordance labels (concordant / in_vivo_specific / in_vitro_specific)

Writes:
  - notebooks/crispr_star/results/agreement_metrics.parquet
  - notebooks/crispr_star/results/discordant_genes.parquet

Usage:
    source scripts/activate_env.sh
    python scripts/elling2024/04_agreement_metrics.py
    python scripts/elling2024/04_agreement_metrics.py --data-dir data/elling2024
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from crispr_al.metrics import compute_cat, compute_jaccard, compute_discordance_labels
from crispr_al.io import load_parquet, save_parquet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data/elling2024")
DEFAULT_RESULTS_DIR = Path("notebooks/crispr_star/results")

CAT_N_VALUES = [10, 20, 50, 100, 200, 500]
NORMZ_HIT_THRESHOLD = 2.0


def pivot_scores(scores_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long table to wide: index=gene_symbol, cols=in_vitro/in_vivo score_norm."""
    wide = scores_long.pivot_table(
        index="gene_symbol", columns="context", values="score_norm", aggfunc="mean"
    )
    wide.columns.name = None
    if "in_vitro" not in wide.columns or "in_vivo" not in wide.columns:
        raise ValueError(
            f"Expected columns 'in_vitro' and 'in_vivo' after pivot; got {list(wide.columns)}"
        )
    wide = wide.dropna(subset=["in_vitro", "in_vivo"])
    log.info("Paired gene universe: %d genes", len(wide))
    return wide


def main(data_dir: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    scores_long = load_parquet(str(data_dir / "scores_long.parquet"))
    wide = pivot_scores(scores_long)

    y_vitro = wide["in_vitro"].values
    y_vivo = wide["in_vivo"].values

    # --- Global agreement ---
    from scipy.stats import pearsonr, spearmanr
    pearson_r, pearson_p = pearsonr(y_vitro, y_vivo)
    spearman_r, spearman_p = spearmanr(y_vitro, y_vivo)
    log.info("Pearson r=%.3f (p=%.2e), Spearman rho=%.3f (p=%.2e)",
             pearson_r, pearson_p, spearman_r, spearman_p)

    # --- CAT@N ---
    cat_rows = []
    for n in CAT_N_VALUES:
        cat_sens = compute_cat(y_vitro, y_vivo, n, direction="sensitiser")
        cat_res = compute_cat(y_vitro, y_vivo, n, direction="resistor")
        cat_rows.append({"n": n, "cat_sensitiser": cat_sens, "cat_resistor": cat_res})
        log.info("CAT@%d: sensitiser=%.3f, resistor=%.3f", n, cat_sens, cat_res)

    # --- Hit overlap Jaccard ---
    sens_vitro = set(wide.index[y_vitro < -NORMZ_HIT_THRESHOLD])
    sens_vivo = set(wide.index[y_vivo < -NORMZ_HIT_THRESHOLD])
    res_vitro = set(wide.index[y_vitro > NORMZ_HIT_THRESHOLD])
    res_vivo = set(wide.index[y_vivo > NORMZ_HIT_THRESHOLD])

    jac_sens = compute_jaccard(sens_vitro, sens_vivo)
    jac_res = compute_jaccard(res_vitro, res_vivo)
    log.info(
        "Jaccard sensitiser=%.3f (%d vitro, %d vivo hits), resistor=%.3f (%d vitro, %d vivo hits)",
        jac_sens, len(sens_vitro), len(sens_vivo),
        jac_res, len(res_vitro), len(res_vivo),
    )

    # --- Assemble global metrics parquet ---
    global_row = {
        "screen_pair": "elling2024_invitro_vs_invivo",
        "n_genes": len(wide),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "jaccard_sensitiser": float(jac_sens),
        "jaccard_resistor": float(jac_res),
        "n_hits_vitro_sens": len(sens_vitro),
        "n_hits_vivo_sens": len(sens_vivo),
        "n_hits_vitro_res": len(res_vitro),
        "n_hits_vivo_res": len(res_vivo),
    }
    for row in cat_rows:
        global_row[f"cat_{row['n']}_sens"] = row["cat_sensitiser"]
        global_row[f"cat_{row['n']}_res"] = row["cat_resistor"]

    agreement_df = pd.DataFrame([global_row])
    cat_df = pd.DataFrame(cat_rows)
    # Store CAT curve in the parquet as a nested column via a join
    agreement_df["cat_curve"] = [cat_df.to_dict("records")]

    save_parquet(agreement_df, str(results_dir / "agreement_metrics.parquet"))
    log.info("Saved agreement_metrics.parquet")

    # --- Per-gene discordance labels ---
    discordant_df = compute_discordance_labels(wide, y_vitro_col="in_vitro", y_vivo_col="in_vivo")
    save_parquet(discordant_df, str(results_dir / "discordant_genes.parquet"))
    label_counts = discordant_df["concordance_label"].value_counts()
    log.info("Concordance labels:\n%s", label_counts.to_string())
    log.info("Saved discordant_genes.parquet")
    log.info("Next step: python scripts/elling2024/05_predictive_model.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Elling2024 data directory")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Results output directory")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.results_dir))
