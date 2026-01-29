#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
from datetime import datetime

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


DEFAULT_GENES = [
    "TP53",
    "EGFR",
    "KRAS",
    "MYC",
    "BRCA1",
    "CDK2",
]


def _write_checksums(root_dir, files):
    import hashlib

    checksum_path = os.path.join(root_dir, "checksums.sha256")
    with open(checksum_path, "w", encoding="utf-8") as handle:
        for rel_path in files:
            full_path = os.path.join(root_dir, rel_path)
            digest = hashlib.sha256()
            with open(full_path, "rb") as data:
                for chunk in iter(lambda: data.read(1024 * 1024), b""):
                    digest.update(chunk)
            handle.write(f"{digest.hexdigest()}  {rel_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Create a tiny GEARS/IterPert-compatible AnnData dataset"
    )
    parser.add_argument("--output-root", default="tests/data")
    parser.add_argument("--dataset-name", default="gears_iterpert_tiny")
    parser.add_argument("--version", default="2026-01-25")
    parser.add_argument("--cell-type", default="K562")
    parser.add_argument("--cells-per-pert", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = args.output_root
    dataset_dir = os.path.join(output_root, args.dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    if os.listdir(dataset_dir) and not args.force:
        raise FileExistsError(
            f"Target directory is not empty: {dataset_dir}. Use --force to overwrite."
        )

    rng = np.random.default_rng(args.seed)
    gene_names = list(DEFAULT_GENES)
    conditions = ["ctrl"] + [f"{gene}+ctrl" for gene in gene_names]

    rows = []
    obs_rows = []
    for condition in conditions:
        for _ in range(args.cells_per_pert):
            expr = rng.normal(loc=1.0, scale=0.35, size=len(gene_names))
            expr = np.clip(expr, 0, None)
            if condition != "ctrl":
                pert_gene = condition.split("+")[0]
                gene_idx = gene_names.index(pert_gene)
                expr[gene_idx] += 1.2
            rows.append(expr)
            dose_val = "1+1" if "+" in condition else "1"
            obs_rows.append(
                {
                    "condition": condition,
                    "cell_type": args.cell_type,
                    "dose_val": dose_val,
                    "control": 0 if "+" in condition else 1,
                    "condition_name": f"{args.cell_type}_{condition}_{dose_val}",
                }
            )

    X = sp.csr_matrix(np.asarray(rows, dtype=np.float32))
    obs = pd.DataFrame(obs_rows)
    var = pd.DataFrame({"gene_name": gene_names}, index=gene_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.var_names = gene_names
    adata.uns["log1p"] = {"base": None}

    h5ad_path = os.path.join(dataset_dir, "perturb_processed.h5ad")
    adata.write_h5ad(h5ad_path)

    metadata = {
        "name": args.dataset_name,
        "version": args.version,
        "genes": gene_names,
        "conditions": conditions,
        "cells_per_pert": args.cells_per_pert,
        "cell_type": args.cell_type,
        "seed": args.seed,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "notes": "Tiny synthetic AnnData for GEARS/IterPert tests",
    }
    metadata_path = os.path.join(dataset_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_checksums(dataset_dir, ["perturb_processed.h5ad", "metadata.json"])

    gene2go = {gene: ["GO:0000001"] for gene in gene_names}
    gene2go_path = os.path.join(output_root, "gene2go_all.pkl")
    with open(gene2go_path, "wb") as handle:
        pickle.dump(gene2go, handle)

    print(f"Dataset written to {dataset_dir}")
    print(f"gene2go_all.pkl written to {gene2go_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
