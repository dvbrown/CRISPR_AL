"""Split generation for Design A (within-screen), Design B (cross-screen), and Olivieri LODO."""
import hashlib
import json

import numpy as np

# Design A constants
SEED_START = 11001
SCREEN_ID = "chen2019_1393"

# Design B constants
XFER_SEED_START = 21001
XFER_SEED_START_REVERSE = 22001

# Shared metadata constants used in split records and schema validation.
_AIM1_VENETOCLAX = "aim1_venetoclax"
_AIM1_TRANSFER = "aim1_transfer"


def _sha256_hex16(payload: dict) -> str:
    """Compute SHA-256 hash (first 16 hex chars) of a JSON-serialised payload."""
    encoded = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def generate_splits(all_genes: list, n_repeats: int = 25, train_size: int = 2000) -> list:
    """Generate n_repeats random gene holdout splits (Design A, within-screen).

    Returns list of dicts with:
      split_id, generator_id, family, aim, metrics_profile, seed,
      repeat_index, train_screen_id, test_screen_id, split_hash,
      train_genes, test_genes
    """
    splits = []
    for i in range(n_repeats):
        seed = SEED_START + i
        rng = np.random.default_rng(seed)
        train_genes = set(rng.choice(all_genes, size=train_size, replace=False))
        test_genes = [g for g in all_genes if g not in train_genes]

        split_hash = compute_split_hash("aim1_random_gene_holdout", SCREEN_ID, seed, train_genes)

        splits.append({
            "split_id": f"aim1_random_{SCREEN_ID}_r{i+1:03d}",
            "generator_id": "aim1_random_gene_holdout",
            "family": "random_gene_holdout",
            "aim": _AIM1_VENETOCLAX,
            "metrics_profile": _AIM1_TRANSFER,
            "seed": seed,
            "repeat_index": i + 1,
            "train_screen_id": SCREEN_ID,
            "test_screen_id": SCREEN_ID,
            "split_hash": split_hash,
            "train_genes": sorted(train_genes),
            "test_genes": sorted(test_genes),
        })
    return splits


def generate_cross_screen_splits(
    train_genes: list,
    test_genes: list,
    train_screen_id: str,
    test_screen_id: str,
    n_repeats: int = 30,
    train_size: int = 2000,
    seed_start: int = XFER_SEED_START,
) -> list:
    """Generate n_repeats cross-screen transfer splits (Design B).

    Samples train_size genes from train_genes. Test set is all test_genes
    not in the sampled train set.

    Returns list of dicts with:
      split_id, generator_id, family, aim, metrics_profile, seed,
      repeat_index, train_screen_id, test_screen_id, split_hash,
      train_genes, test_genes
    """
    train_genes = list(train_genes)
    test_genes = list(test_genes)
    splits = []
    for i in range(n_repeats):
        seed = seed_start + i
        rng = np.random.default_rng(seed)
        sampled_train = set(rng.choice(train_genes, size=train_size, replace=False))
        split_test = [g for g in test_genes if g not in sampled_train]

        assert sampled_train & set(split_test) == set(), (
            f"Leakage detected in split {i+1}: train/test gene overlap"
        )

        split_hash = compute_split_hash(
            "aim1_cross_screen_transfer",
            f"{train_screen_id}_to_{test_screen_id}",
            seed,
            sampled_train,
        )

        splits.append({
            "split_id": f"aim1_xfer_{train_screen_id}_to_{test_screen_id}_r{i+1:03d}",
            "generator_id": "aim1_cross_screen_transfer",
            "family": "context_zeroshot",
            "aim": _AIM1_VENETOCLAX,
            "metrics_profile": _AIM1_TRANSFER,
            "seed": seed,
            "repeat_index": i + 1,
            "train_screen_id": train_screen_id,
            "test_screen_id": test_screen_id,
            "split_hash": split_hash,
            "train_genes": sorted(sampled_train),
            "test_genes": sorted(split_test),
        })
    return splits


def compute_split_hash(generator_id: str, screen_id: str, seed: int, train_genes) -> str:
    """Compute reproducible SHA-256 hash (first 16 hex chars) for a split."""
    return _sha256_hex16({
        "generator_id": generator_id,
        "screen_id": screen_id,
        "seed": seed,
        "train_genes": sorted(train_genes),
    })


def compute_overlap_baseline_hash(
    generator_id: str,
    train_screen_id: str,
    test_screen_id: str,
    shared_genes: list,
) -> str:
    """Compute deterministic SHA-256 hash for the overlap-only baseline record.

    Encodes generator_id, ordered screen IDs, and sorted shared gene list.
    Uses seed=0 by convention.
    """
    return _sha256_hex16({
        "generator_id": generator_id,
        "train_screen_id": train_screen_id,
        "test_screen_id": test_screen_id,
        "seed": 0,
        "shared_genes": sorted(shared_genes),
    })


# ── Olivieri 2020 LODO splits ─────────────────────────────────────────────────

def generate_lodo_splits(
    normz_matrix,
    screen_metadata,
    library: str,
) -> list:
    """Generate leave-one-drug-out split configurations within a library.

    Parameters
    ----------
    normz_matrix    : DataFrame, genes × screens (index=gene_symbol, columns=screen_label)
    screen_metadata : DataFrame with columns screen_label and library
    library         : 'TKOv2' or 'TKOv3'

    Returns list of dicts with keys: test_screen, train_screens.
    Only includes screens present in both metadata and normz_matrix columns.
    """
    lib_screens = screen_metadata.loc[
        screen_metadata["library"] == library, "screen_label"
    ].tolist()
    lib_screens = [s for s in lib_screens if s in normz_matrix.columns]
    return [
        {"test_screen": s, "train_screens": [t for t in lib_screens if t != s]}
        for s in lib_screens
    ]


# ── Aim 1 EuMyc split functions ───────────────────────────────────────────────

# Hallmark gene sets used as seeds for nutlin-3a / p53 screens
APOPTOSIS_P53_HALLMARKS = [
    "HALLMARK_APOPTOSIS",
    "HALLMARK_P53_PATHWAY",
    "HALLMARK_DNA_REPAIR",
    "HALLMARK_G2M_CHECKPOINT",
]

# Hallmark gene sets used as seeds for S63845 (MCL-1 inhibitor) screens
BCL2_APOPTOSIS_HALLMARKS = [
    "HALLMARK_APOPTOSIS",
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
    "HALLMARK_PI3K_AKT_MTOR_SIGNALING",
    "HALLMARK_MYC_TARGETS_V1",
]

# Reactome pathway IDs for apoptosis / p53 regulation (human IDs)
REACTOME_APOPTOSIS_IDS = [
    "R-HSA-109581",   # Apoptosis
    "R-HSA-5633007",  # Regulation of TP53 Expression and Degradation
    "R-HSA-3700989",  # Transcriptional Regulation by Small Molecules (p53)
    "R-HSA-6796648",  # TP53 Regulates Transcription of Genes Involved in G1 Cell Cycle Arrest
    "R-HSA-111452",   # Activation of BH3-only proteins
    "R-HSA-114452",   # Activation of BIM and translocation to mitochondria
    "R-HSA-139915",   # Activation of NOXA and translocation to mitochondria
    "R-HSA-5357769",  # BCL-2 family members and regulation of apoptosis
]


def split_random(
    gene_list: list,
    n_train: int,
    seed: int,
) -> tuple:
    """Randomly partition genes into training and holdout sets.

    Parameters
    ----------
    gene_list : list of gene symbols (must be a list for index stability)
    n_train   : number of training genes
    seed      : RNG seed for reproducibility

    Returns
    -------
    (train_genes, holdout_genes)
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(gene_list))
    return [gene_list[i] for i in idx[:n_train]], \
           [gene_list[i] for i in idx[n_train:]]


def split_reactome_stratified(
    gene_list: list,
    reactome_membership: dict,
    n_train: int,
    seed: int,
    min_pathway_size: int = 10,
    max_pathway_size: int = 500,
) -> tuple:
    """Reactome-stratified split: 1 gene per qualifying pathway, random fill.

    Parameters
    ----------
    gene_list          : list of gene symbols
    reactome_membership: pathway_id → list of gene symbols
    n_train            : total training budget
    seed               : RNG seed
    min_pathway_size   : exclude pathways smaller than this (after intersection)
    max_pathway_size   : exclude pathways larger than this (after intersection)

    Returns
    -------
    (train_genes, holdout_genes)
    """
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)

    qualifying = {
        pid: [g for g in genes if g in gene_set]
        for pid, genes in reactome_membership.items()
        if min_pathway_size <= len([g for g in genes if g in gene_set]) <= max_pathway_size
    }

    pathway_ids = sorted(qualifying.keys())
    rng.shuffle(pathway_ids)

    selected = set()
    for pid in pathway_ids:
        candidates = [g for g in qualifying[pid] if g not in selected]
        if candidates:
            chosen = rng.choice(candidates)
            selected.add(str(chosen))
        if len(selected) >= n_train:
            break

    remaining = [g for g in gene_list if g not in selected]
    n_fill = n_train - len(selected)
    if n_fill > 0:
        fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
        selected.update(fill)

    train_genes = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes


def split_hallmark_stratified(
    gene_list: list,
    hallmark_membership: dict,
    n_train: int,
    seed: int,
    n_per_hallmark: int = 5,
) -> tuple:
    """Hallmark-stratified split: n_per_hallmark genes per gene set, random fill.

    Parameters
    ----------
    gene_list          : list of gene symbols
    hallmark_membership: hallmark_name → list of gene symbols
    n_train            : total training budget
    seed               : RNG seed
    n_per_hallmark     : genes sampled per Hallmark gene set

    Returns
    -------
    (train_genes, holdout_genes)
    """
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)
    selected = set()

    hallmark_names = sorted(hallmark_membership.keys())
    rng.shuffle(hallmark_names)

    for hname in hallmark_names:
        candidates = [g for g in hallmark_membership[hname]
                      if g in gene_set and g not in selected]
        n_sample = min(n_per_hallmark, len(candidates))
        if n_sample > 0:
            chosen = rng.choice(candidates, size=n_sample, replace=False)
            selected.update(str(g) for g in chosen)

    remaining = [g for g in gene_list if g not in selected]
    n_fill = n_train - len(selected)
    if n_fill > 0:
        fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
        selected.update(fill)

    train_genes = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes


def split_apoptosis_p53_seeded(
    gene_list: list,
    hallmark_membership: dict,
    n_train: int,
    seed: int,
    seed_hallmarks: list = None,
    max_seed_fraction: float = 0.20,
) -> tuple:
    """Apoptosis/p53 Hallmark genes seeded, remainder filled randomly.

    Fixes a core set of genes from apoptosis and p53 pathway Hallmark gene
    sets as mandatory training genes (capped at max_seed_fraction × n_train),
    then fills the remaining budget randomly.

    Parameters
    ----------
    gene_list          : list of gene symbols
    hallmark_membership: hallmark_name → list of gene symbols
    n_train            : total training budget
    seed               : RNG seed
    seed_hallmarks     : Hallmark names to use as seeds (default: APOPTOSIS_P53_HALLMARKS)
    max_seed_fraction  : cap seed genes at this fraction of n_train

    Returns
    -------
    (train_genes, holdout_genes)
    """
    if seed_hallmarks is None:
        seed_hallmarks = APOPTOSIS_P53_HALLMARKS
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)
    max_seeds = int(n_train * max_seed_fraction)

    seed_genes = set()
    for hname in seed_hallmarks:
        seed_genes.update(g for g in hallmark_membership.get(hname, [])
                          if g in gene_set)

    if len(seed_genes) > max_seeds:
        seed_genes = set(str(g) for g in rng.choice(sorted(seed_genes),
                                                     size=max_seeds, replace=False))

    remaining = [g for g in gene_list if g not in seed_genes]
    n_fill = n_train - len(seed_genes)
    fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
    selected = seed_genes | set(fill)

    train_genes = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes


def split_bcl2_seeded(
    gene_list: list,
    hallmark_membership: dict,
    n_train: int,
    seed: int,
    seed_hallmarks: list = None,
    max_seed_fraction: float = 0.20,
) -> tuple:
    """BCL-2/survival Hallmark genes seeded, remainder filled randomly.

    Same logic as split_apoptosis_p53_seeded but with BCL2_APOPTOSIS_HALLMARKS
    as the default seed gene sets (relevant for S63845/MCL-1 inhibitor arms).

    Returns
    -------
    (train_genes, holdout_genes)
    """
    if seed_hallmarks is None:
        seed_hallmarks = BCL2_APOPTOSIS_HALLMARKS
    return split_apoptosis_p53_seeded(
        gene_list, hallmark_membership, n_train, seed,
        seed_hallmarks=seed_hallmarks,
        max_seed_fraction=max_seed_fraction,
    )


def split_reactome_apoptosis_oversampled(
    gene_list: list,
    reactome_membership: dict,
    n_train: int,
    seed: int,
    apoptosis_pathway_ids: list = None,
    n_per_apoptosis: int = 5,
    n_per_other: int = 1,
    min_pathway_size: int = 10,
    max_pathway_size: int = 500,
) -> tuple:
    """Reactome-stratified with oversampling of apoptosis/p53 pathways.

    For apoptosis/p53 Reactome pathways: sample n_per_apoptosis genes each.
    For all other qualifying pathways: sample n_per_other gene each.
    Remaining budget filled randomly.

    Parameters
    ----------
    gene_list            : list of gene symbols
    reactome_membership  : pathway_id → list of gene symbols
    n_train              : total training budget
    seed                 : RNG seed
    apoptosis_pathway_ids: Reactome pathway IDs to oversample (default: REACTOME_APOPTOSIS_IDS)
    n_per_apoptosis      : genes sampled per apoptosis pathway
    n_per_other          : genes sampled per non-apoptosis qualifying pathway
    min_pathway_size     : minimum pathway size (after gene_list intersection)
    max_pathway_size     : maximum pathway size (after gene_list intersection)

    Returns
    -------
    (train_genes, holdout_genes)
    """
    if apoptosis_pathway_ids is None:
        apoptosis_pathway_ids = REACTOME_APOPTOSIS_IDS
    apoptosis_set = set(apoptosis_pathway_ids)
    rng = np.random.default_rng(seed)
    gene_set = set(gene_list)
    selected = set()

    for pid, genes in reactome_membership.items():
        members = [g for g in genes if g in gene_set and g not in selected]
        pathway_size = len([g for g in genes if g in gene_set])
        if not (min_pathway_size <= pathway_size <= max_pathway_size):
            continue

        n_sample = n_per_apoptosis if pid in apoptosis_set else n_per_other
        n_sample = min(n_sample, len(members))
        if n_sample > 0:
            chosen = rng.choice(members, size=n_sample, replace=False)
            selected.update(str(g) for g in chosen)

        if len(selected) >= n_train:
            break

    remaining = [g for g in gene_list if g not in selected]
    n_fill = n_train - len(selected)
    if n_fill > 0:
        fill = rng.choice(remaining, size=n_fill, replace=False).tolist()
        selected.update(fill)

    train_genes = [g for g in gene_list if g in selected]
    holdout_genes = [g for g in gene_list if g not in selected]
    return train_genes, holdout_genes
