#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ModelConfig:
    hidden_dim: int = 128
    residual_dim: int = 64
    dropout: float = 0.1
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    ensemble_size: int = 5
    activation: str = "relu"


@dataclass
class ALConfig:
    n_rounds: int = 5
    n_init: int = 200
    n_query: int = 200
    beta: float = 1.0
    pool_multiplier: int = 5
    diversity: str = "kcenter"
    hit_threshold: float = 3.0
    top_k: int = 50
    seed: int = 7


class EpistasisSurrogate(nn.Module):
    def __init__(self, hidden_dim, residual_dim, dropout, activation):
        super().__init__()
        self.encoder = nn.Linear(2, hidden_dim)
        if activation == "gelu":
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        self.additive_proj = nn.Linear(hidden_dim, 1)
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, residual_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(residual_dim, 1),
        )

    def forward(self, x):
        hidden = self.activation(self.encoder(x))
        additive = self.additive_proj(hidden)
        residual = self.residual(hidden)
        return additive + residual, additive, hidden


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_sep(path, sep):
    if sep:
        return sep
    suffix = Path(path).suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return "\t"
    if suffix in {".csv"}:
        return ","
    return None


def read_table(path, sep=None):
    sep = infer_sep(path, sep)
    if sep is None:
        return pd.read_csv(path, sep=None, engine="python")
    return pd.read_csv(path, sep=sep)


def load_single_gene_effects(
    path, gene_col, effect_col, cell_line_col=None, cell_line=None, sep=None
):
    df = read_table(path, sep=sep)
    missing = [col for col in [gene_col, effect_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in DepMap table: {missing}")
    df = df.copy()
    df[gene_col] = df[gene_col].astype(str).str.strip()
    df[effect_col] = pd.to_numeric(df[effect_col], errors="coerce")
    if cell_line_col and cell_line:
        if cell_line_col not in df.columns:
            raise ValueError(f"Cell line column '{cell_line_col}' not in DepMap table")
        df = df[df[cell_line_col].astype(str).str.upper() == cell_line.upper()]
    df = df.dropna(subset=[gene_col, effect_col])
    df = df[[gene_col, effect_col]].rename(
        columns={gene_col: "gene_name", effect_col: "effect"}
    )
    df = df.drop_duplicates(subset=["gene_name"], keep="first")
    return df


def canonicalize_pairs(df, gene_a_col, gene_b_col):
    gene_a_raw = df[gene_a_col].astype(str).str.strip()
    gene_b_raw = df[gene_b_col].astype(str).str.strip()
    swap = gene_a_raw > gene_b_raw
    gene_a = gene_a_raw.where(~swap, gene_b_raw)
    gene_b = gene_b_raw.where(~swap, gene_a_raw)
    df = df.copy()
    df["gene_a"] = gene_a
    df["gene_b"] = gene_b
    df = df[df["gene_a"] != df["gene_b"]]
    return df


def load_gi_map(path, gene_a_col, gene_b_col, gi_col, sep=None):
    df = read_table(path, sep=sep)
    missing = [col for col in [gene_a_col, gene_b_col, gi_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in GI map: {missing}")
    df = df.copy()
    df = canonicalize_pairs(df, gene_a_col, gene_b_col)
    df[gi_col] = pd.to_numeric(df[gi_col], errors="coerce")
    df = df.dropna(subset=["gene_a", "gene_b", gi_col])
    df = df.drop_duplicates(subset=["gene_a", "gene_b"], keep="first")
    df = df[["gene_a", "gene_b", gi_col]].rename(columns={gi_col: "gi_score"})
    return df


def prepare_candidate_pairs(effect_df, gi_df):
    effect_map = dict(zip(effect_df["gene_name"], effect_df["effect"]))
    gi_df = gi_df[
        gi_df["gene_a"].isin(effect_map) & gi_df["gene_b"].isin(effect_map)
    ].copy()
    gi_df["effect_a"] = gi_df["gene_a"].map(effect_map)
    gi_df["effect_b"] = gi_df["gene_b"].map(effect_map)
    gi_df = gi_df.dropna(subset=["effect_a", "effect_b", "gi_score"])
    features = gi_df[["effect_a", "effect_b"]].to_numpy(dtype=np.float32)
    gi_scores = gi_df["gi_score"].to_numpy(dtype=np.float32)
    linear_baseline = (
        gi_df[["effect_a", "effect_b"]].sum(axis=1).to_numpy(dtype=np.float32)
    )
    return gi_df.reset_index(drop=True), features, gi_scores, linear_baseline


def train_single_model(x, y, config, seed, device):
    torch.manual_seed(seed)
    model = EpistasisSurrogate(
        hidden_dim=config.hidden_dim,
        residual_dim=config.residual_dim,
        dropout=config.dropout,
        activation=config.activation,
    ).to(device)
    dataset = TensorDataset(
        torch.from_numpy(x).float(),
        torch.from_numpy(y).float(),
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(config.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds, _, _ = model(xb)
            loss = loss_fn(preds.squeeze(-1), yb)
            loss.backward()
            optimizer.step()
    return model


def train_ensemble(x, y, config, seed, device):
    models = []
    for i in range(config.ensemble_size):
        model_seed = seed + i * 13
        models.append(train_single_model(x, y, config, model_seed, device))
    return models


def predict_with_model(model, x, batch_size, device):
    model.eval()
    preds = []
    additive = []
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            pred, add, emb = model(xb)
            preds.append(pred.squeeze(-1).cpu().numpy())
            additive.append(add.squeeze(-1).cpu().numpy())
            embeddings.append(emb.cpu().numpy())
    return (
        np.concatenate(preds, axis=0),
        np.concatenate(additive, axis=0),
        np.concatenate(embeddings, axis=0),
    )


def predict_ensemble(models, x, batch_size, device):
    preds = []
    additives = []
    embeddings = []
    for model in models:
        pred, add, emb = predict_with_model(model, x, batch_size, device)
        preds.append(pred)
        additives.append(add)
        embeddings.append(emb)
    preds = np.stack(preds, axis=0)
    additives = np.stack(additives, axis=0)
    embeddings = np.stack(embeddings, axis=0)
    mu = preds.mean(axis=0)
    sigma = preds.std(axis=0)
    additive_mu = additives.mean(axis=0)
    embedding_mu = embeddings.mean(axis=0)
    return mu, sigma, additive_mu, embedding_mu


def safe_corr(a, b):
    if len(a) < 2:
        return float("nan")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def kcenter_select(candidate_embeddings, labeled_embeddings, n_select, rng):
    if n_select <= 0 or candidate_embeddings.size == 0:
        return []
    if labeled_embeddings is None or len(labeled_embeddings) == 0:
        start_idx = int(rng.integers(candidate_embeddings.shape[0]))
        selected = [start_idx]
        min_dist = pairwise_distances(
            candidate_embeddings, candidate_embeddings[[start_idx]]
        ).ravel()
    else:
        min_dist = pairwise_distances(candidate_embeddings, labeled_embeddings).min(
            axis=1
        )
        selected = []
    while len(selected) < n_select:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        new_dist = pairwise_distances(
            candidate_embeddings, candidate_embeddings[[idx]]
        ).ravel()
        min_dist = np.minimum(min_dist, new_dist)
        min_dist[idx] = -np.inf
    return selected


def kmeans_select(embeddings, n_select, seed):
    if n_select <= 0:
        return []
    if n_select >= len(embeddings):
        return list(range(len(embeddings)))
    kmeans = KMeans(n_clusters=n_select, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(embeddings)
    centers = kmeans.cluster_centers_
    selected = []
    for cluster_id in range(n_select):
        cluster_idx = np.where(labels == cluster_id)[0]
        if len(cluster_idx) == 0:
            continue
        dist = np.linalg.norm(embeddings[cluster_idx] - centers[cluster_id], axis=1)
        selected.append(int(cluster_idx[np.argmin(dist)]))
    return selected


def typiclust_select(
    candidate_embeddings, labeled_embeddings, n_select, seed, n_neighbors=10
):
    if n_select <= 0:
        return []
    if n_select >= len(candidate_embeddings):
        return list(range(len(candidate_embeddings)))
    extra_clusters = 0 if labeled_embeddings is None else len(labeled_embeddings)
    n_clusters = min(len(candidate_embeddings), n_select + extra_clusters)
    if n_clusters <= 1:
        return kmeans_select(candidate_embeddings, n_select, seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
    labels = kmeans.fit_predict(candidate_embeddings)
    labeled_counts = np.zeros(n_clusters, dtype=int)
    if labeled_embeddings is not None and len(labeled_embeddings) > 0:
        labeled_labels = kmeans.predict(labeled_embeddings)
        for label in labeled_labels:
            labeled_counts[label] += 1
    cluster_sizes = np.bincount(labels, minlength=n_clusters)
    cluster_order = sorted(
        range(n_clusters),
        key=lambda idx: (labeled_counts[idx], -cluster_sizes[idx]),
    )
    selected = []
    for cluster_id in cluster_order:
        if len(selected) >= n_select:
            break
        cluster_idx = np.where(labels == cluster_id)[0]
        if len(cluster_idx) == 0:
            continue
        k = min(n_neighbors, len(cluster_idx))
        neighbors = NearestNeighbors(n_neighbors=k)
        neighbors.fit(candidate_embeddings[cluster_idx])
        distances, _ = neighbors.kneighbors(candidate_embeddings[cluster_idx])
        typicality = distances.mean(axis=1)
        selected.append(int(cluster_idx[np.argmin(typicality)]))
    if len(selected) < n_select:
        remaining = [
            idx for idx in range(len(candidate_embeddings)) if idx not in selected
        ]
        rng = np.random.default_rng(seed)
        extra = rng.choice(
            remaining, size=min(n_select - len(selected), len(remaining)), replace=False
        )
        selected.extend([int(idx) for idx in extra])
    return selected


def select_batch(strategy, embeddings, labeled_embeddings, n_select, seed):
    rng = np.random.default_rng(seed)
    if n_select <= 0:
        return []
    if strategy == "random":
        return [
            int(idx)
            for idx in rng.choice(len(embeddings), size=n_select, replace=False)
        ]
    if strategy == "kmeans":
        return kmeans_select(embeddings, n_select, seed)
    if strategy == "typiclust":
        return typiclust_select(embeddings, labeled_embeddings, n_select, seed)
    return kcenter_select(embeddings, labeled_embeddings, n_select, rng)


def compute_round_metrics(
    round_idx,
    labeled_idx,
    selected_idx,
    gi_scores,
    mu_all,
    additive_mu,
    linear_baseline,
    hit_threshold,
    top_k,
):
    metrics = []

    def add_metric(name, value):
        metrics.append({"round": round_idx, "metric": name, "value": value})

    selected_gi = gi_scores[selected_idx]
    labeled_gi = gi_scores[labeled_idx]
    hit_rate_batch = (
        float(np.mean(np.abs(selected_gi) >= hit_threshold))
        if len(selected_gi)
        else float("nan")
    )
    hit_rate_labeled = (
        float(np.mean(np.abs(labeled_gi) >= hit_threshold))
        if len(labeled_gi)
        else float("nan")
    )
    add_metric("hit_rate_batch", hit_rate_batch)
    add_metric("hit_rate_labeled", hit_rate_labeled)
    add_metric(
        "mean_abs_gi_batch",
        float(np.mean(np.abs(selected_gi))) if len(selected_gi) else float("nan"),
    )
    add_metric("corr_mu_oracle", safe_corr(mu_all, gi_scores))
    add_metric("corr_additive_linear", safe_corr(additive_mu, linear_baseline))

    top_k = min(top_k, len(gi_scores))
    top_idx = np.argsort(-np.abs(mu_all))[:top_k]
    top_hit_rate = (
        float(np.mean(np.abs(gi_scores[top_idx]) >= hit_threshold))
        if top_k
        else float("nan")
    )
    add_metric("topk_hit_rate", top_hit_rate)

    summary = {
        "round": round_idx,
        "n_labeled": len(labeled_idx),
        "n_query": len(selected_idx),
        "hit_rate_batch": hit_rate_batch,
        "hit_rate_labeled": hit_rate_labeled,
        "corr_mu_oracle": safe_corr(mu_all, gi_scores),
        "topk_hit_rate": top_hit_rate,
    }
    return summary, metrics


def run_active_learning(
    pairs_df,
    features,
    gi_scores,
    linear_baseline,
    model_config,
    al_config,
    output_dir,
    device,
):
    rng = np.random.default_rng(al_config.seed)
    n_pairs = len(pairs_df)
    if al_config.n_init >= n_pairs:
        raise ValueError("n_init must be smaller than number of candidate pairs")
    labeled_idx = rng.choice(n_pairs, size=al_config.n_init, replace=False)
    unlabeled_idx = np.setdiff1d(np.arange(n_pairs), labeled_idx)

    round_summaries = []
    metrics_long = []
    selections = []

    for round_idx in range(al_config.n_rounds):
        print(f"Round {round_idx}: training on {len(labeled_idx)} labeled pairs")
        models = train_ensemble(
            features[labeled_idx],
            gi_scores[labeled_idx],
            model_config,
            al_config.seed + round_idx * 101,
            device,
        )
        mu_all, sigma_all, additive_mu, embeddings_all = predict_ensemble(
            models, features, model_config.batch_size, device
        )
        if len(unlabeled_idx) == 0:
            print("No unlabeled pairs remaining. Stopping.")
            break

        acquisition = (
            np.abs(mu_all[unlabeled_idx]) + al_config.beta * sigma_all[unlabeled_idx]
        )
        pool_size = min(
            len(unlabeled_idx), al_config.pool_multiplier * al_config.n_query
        )
        pool_order = np.argsort(-acquisition)[:pool_size]
        pool_idx = unlabeled_idx[pool_order]

        pool_embeddings = embeddings_all[pool_idx]
        labeled_embeddings = embeddings_all[labeled_idx] if len(labeled_idx) else None
        n_query = min(al_config.n_query, len(pool_idx))
        selected_rel = select_batch(
            al_config.diversity,
            pool_embeddings,
            labeled_embeddings,
            n_query,
            al_config.seed + round_idx * 17,
        )
        selected_idx = pool_idx[selected_rel]

        for idx in selected_idx:
            selections.append(
                {
                    "round": round_idx,
                    "gene_a": pairs_df.loc[idx, "gene_a"],
                    "gene_b": pairs_df.loc[idx, "gene_b"],
                    "gi_score": float(gi_scores[idx]),
                    "mu": float(mu_all[idx]),
                    "sigma": float(sigma_all[idx]),
                    "acquisition": float(
                        np.abs(mu_all[idx]) + al_config.beta * sigma_all[idx]
                    ),
                }
            )

        summary, metrics = compute_round_metrics(
            round_idx,
            labeled_idx,
            selected_idx,
            gi_scores,
            mu_all,
            additive_mu,
            linear_baseline,
            al_config.hit_threshold,
            al_config.top_k,
        )
        round_summaries.append(summary)
        metrics_long.extend(metrics)

        labeled_idx = np.concatenate([labeled_idx, selected_idx])
        unlabeled_idx = np.setdiff1d(unlabeled_idx, selected_idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    rounds_path = output_dir / "rounds.csv"
    metrics_path = output_dir / "metrics.csv"
    selections_path = output_dir / "selected_pairs.csv"

    pd.DataFrame(round_summaries).to_csv(rounds_path, index=False)
    pd.DataFrame(metrics_long).to_csv(metrics_path, index=False)
    pd.DataFrame(selections).to_csv(selections_path, index=False)

    print(f"Saved rounds to {rounds_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved selections to {selections_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Active learning for bulk CRISPR epistasis"
    )
    parser.add_argument(
        "--depmap-file", required=True, help="Path to DepMap single-gene effect table"
    )
    parser.add_argument(
        "--gi-file", required=True, help="Path to Horlbeck GI map table"
    )
    parser.add_argument("--depmap-gene-col", default="gene_name")
    parser.add_argument("--depmap-effect-col", default="effect")
    parser.add_argument("--depmap-cell-line-col")
    parser.add_argument("--depmap-cell-line", default="K562")
    parser.add_argument("--gi-gene-a-col", default="gene_a")
    parser.add_argument("--gi-gene-b-col", default="gene_b")
    parser.add_argument("--gi-score-col", default="gi_score")
    parser.add_argument("--depmap-sep")
    parser.add_argument("--gi-sep")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-rounds", type=int, default=5)
    parser.add_argument("--n-init", type=int, default=200)
    parser.add_argument("--n-query", type=int, default=200)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--pool-multiplier", type=int, default=5)
    parser.add_argument(
        "--diversity",
        choices=["kcenter", "kmeans", "typiclust", "random"],
        default="kcenter",
    )
    parser.add_argument("--hit-threshold", type=float, default=3.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--residual-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--activation", choices=["relu", "gelu"], default="relu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)

    effect_df = load_single_gene_effects(
        args.depmap_file,
        gene_col=args.depmap_gene_col,
        effect_col=args.depmap_effect_col,
        cell_line_col=args.depmap_cell_line_col,
        cell_line=args.depmap_cell_line,
        sep=args.depmap_sep,
    )
    gi_df = load_gi_map(
        args.gi_file,
        gene_a_col=args.gi_gene_a_col,
        gene_b_col=args.gi_gene_b_col,
        gi_col=args.gi_score_col,
        sep=args.gi_sep,
    )
    pairs_df, features, gi_scores, linear_baseline = prepare_candidate_pairs(
        effect_df, gi_df
    )
    if len(pairs_df) == 0:
        raise ValueError("No candidate pairs remain after intersection")

    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        residual_dim=args.residual_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        ensemble_size=args.ensemble_size,
        activation=args.activation,
    )
    al_config = ALConfig(
        n_rounds=args.n_rounds,
        n_init=args.n_init,
        n_query=args.n_query,
        beta=args.beta,
        pool_multiplier=args.pool_multiplier,
        diversity=args.diversity,
        hit_threshold=args.hit_threshold,
        top_k=args.top_k,
        seed=args.seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "run_config.json"
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(
            {"model": model_config.__dict__, "active_learning": al_config.__dict__},
            handle,
            indent=2,
        )

    print(f"Loaded {len(pairs_df)} candidate pairs")
    run_active_learning(
        pairs_df,
        features,
        gi_scores,
        linear_baseline,
        model_config,
        al_config,
        output_dir,
        device,
    )


if __name__ == "__main__":
    main()
