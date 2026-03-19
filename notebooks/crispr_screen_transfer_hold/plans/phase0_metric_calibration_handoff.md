# Phase 0 Handoff: Metric Calibration Framework
## CRISPR Active Learning — `dvbrown/CRISPR_AL`

---

## Context

This is the first task in a 5-phase project to build an active learning CRISPR screen pipeline. The project predicts venetoclax resistance/sensitivity from partial CRISPR screens in AML cell lines (Chen 2019 MOLM-13, Sharon 2019 MOLM-13-R1).

**Why this phase comes first:** A recent benchmark debate (Ahlmann-Eltze et al. Nat Methods 2025 vs Miller et al. biorXiv 2025) showed that standard metrics like MSE and Pearson(Δctrl) are often poorly calibrated — a "predict the mean" baseline can appear competitive simply because ~95% of genes are unaffected by perturbation. Before investing in modeling, we must verify our metrics can actually detect signal. This phase implements that verification.

**Key insight for fitness screens vs transcriptomics:** Our screens measure CRISPR fitness effects (CRISPR Score / LFC), not transcriptomics. Most genes in a fitness screen DO have non-zero effects, so the null-perturbation dilution problem is less severe. However, we still need to verify this empirically rather than assume it.

---

## Repository State

**Repo:** `https://github.com/dvbrown/CRISPR_AL`

**Relevant existing code:**

- `src/crispr_al/metrics.py` — already has `compute_regression_metrics`, `compute_ranking_metrics`, `compute_classification_metrics`, `bootstrap_ci_bca`, `flatten_metrics_row`, `build_metrics_record`. Does NOT yet have DRF or calibration functions.
- `src/crispr_al/screen.py` — `load_screen_scores` (Chen), `load_sharon_screen_scores` (Sharon), `zscore_normalize`, `assign_hit_labels_zscore`
- `src/crispr_al/io.py` — `save_parquet`, `load_parquet`, `save_metrics_json`
- `src/crispr_al/plotting.py` — `theme_publication`, `PUBLICATION_COLORS`, `scale_color_publication`, `scale_fill_publication`
- `notebooks/crispr_screen_transfer/design_a_analysis.py` — existing Marimo notebook for Design A (reference for style)

**Data on disk** (all under `data/bulk/`):
```
chen2019_venetoclax/BIOGRID-ORCS-SCREEN_1392-*.txt   # Chen 2019 screen
chen2019_venetoclax/BIOGRID-ORCS-SCREEN_1393-*.txt   # Chen 2019 screen (primary)
sharon2019_venetoclax/BIOGRID-ORCS-SCREEN_1401-*.txt # Sharon 2019 screen (primary)
sharon2019_venetoclax/BIOGRID-ORCS-SCREEN_1402-*.txt
```

**Gene counts (verified):**
- Chen 2019: 19,109 unique genes
- Sharon 2019: 17,230 unique genes
- Shared genes: 17,091

---

## Task: What to Build

### 1. Add `compute_drf()` to `src/crispr_al/metrics.py`

Implement the Dynamic Range Fraction (DRF) from Miller et al. 2025. This is a meta-metric that measures whether a benchmarking metric can distinguish signal from noise.

```python
def compute_drf(
    y_true: np.ndarray,
    y_positive_control: np.ndarray,
    y_negative_control: np.ndarray,
    metric_fn,  # callable: (y_true, y_pred) -> float, higher=better
    epsilon: float = 1e-8,
) -> float:
    """
    Dynamic Range Fraction: measures metric calibration.

    DRF = (metric(y_positive_control) - metric(y_negative_control))
          / (perfect_score - metric(y_negative_control) + epsilon)

    A well-calibrated metric has DRF close to 1.0.
    A poorly calibrated metric has DRF close to 0.0 (positive control
    looks no better than negative control).

    Args:
        y_true: Ground truth scores (n_genes,)
        y_positive_control: Predictions from positive control (e.g. split-half)
        y_negative_control: Predictions from negative control (e.g. global mean)
        metric_fn: Function (y_true, y_pred) -> float. Must be higher-is-better.
                   For Spearman: lambda y, yp: spearmanr(y, yp).statistic
                   For Pearson:  lambda y, yp: pearsonr(y, yp).statistic
                   For neg-RMSE: lambda y, yp: -mean_squared_error(y, yp)**0.5
        epsilon: Numerical stability constant.

    Returns:
        DRF value in [0, 1] (clipped). Values near 1 = well calibrated.
    """
```

Also add a convenience function:

```python
def compute_calibration_report(
    y_true: np.ndarray,
    y_positive_control: np.ndarray,
    y_negative_control: np.ndarray,
) -> dict:
    """
    Compute DRF for all standard metrics used in this project.

    Returns dict with keys:
      drf_spearman, drf_pearson, drf_neg_rmse,
      drf_auroc_sensitizer, drf_precision_at_50, drf_precision_at_100,
      drf_precision_at_200, drf_precision_at_500

    Also includes the raw metric values for positive control, negative control,
    and perfect score for each metric (for debugging).
    """
```

For AUROC and Precision@K, you need hit labels. Add an overloaded version:

```python
def compute_calibration_report_with_hits(
    y_true: np.ndarray,
    y_positive_control: np.ndarray,
    y_negative_control: np.ndarray,
    hit_sensitizer: np.ndarray,  # boolean array
    hit_resistor: np.ndarray,    # boolean array
) -> dict:
    """Full calibration report including ranking and classification metrics."""
```

### 2. Implement Positive and Negative Controls

The controls are computed from the screen data itself (no model needed):

**Negative control — global mean predictor:**
```python
def make_negative_control(y_true: np.ndarray) -> np.ndarray:
    """Returns array of global mean, same shape as y_true."""
    return np.full_like(y_true, fill_value=y_true.mean())
```

**Positive control — split-half (technical duplicate):**
```python
def make_positive_control_split_half(
    screen_df: pd.DataFrame,
    score_col: str = "score_norm",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split genes into two halves. Use half-A scores to predict half-B scores.

    This is the ceiling: if you had a perfect replicate experiment, this is
    how well you could predict it. Any model should aim to approach this.

    Returns:
        (y_true_half_b, y_pred_from_half_a)
        where y_pred_from_half_a[i] = mean score of gene i's half-A counterpart.

    For a fitness screen, this means: randomly assign each gene to half-A or
    half-B, then use half-A's score as the prediction for half-B's score.
    Since each gene has a single score (not per-cell), we implement this as:
    - Randomly permute genes
    - First half: training set (half-A)
    - Second half: test set (half-B)
    - Prediction for each half-B gene = score of a randomly matched half-A gene
      (this simulates what a perfect replicate would look like)

    Note: For fitness screens (unlike Perturb-seq), each gene has one score,
    not a distribution of cells. The split-half positive control therefore
    uses cross-screen replication (Chen vs Sharon overlap) as the true
    positive control — see make_positive_control_cross_screen().
    """
```

**Better positive control for fitness screens — cross-screen replication:**
```python
def make_positive_control_cross_screen(
    chen_scores: pd.Series,   # indexed by gene_symbol
    sharon_scores: pd.Series, # indexed by gene_symbol
) -> tuple[np.ndarray, np.ndarray]:
    """
    Use Chen 2019 scores to predict Sharon 2019 scores on shared genes.

    This is the empirical ceiling for cross-screen transfer: how well does
    one biological replicate predict another? Any model trained on Chen data
    should not be expected to exceed this ceiling on Sharon data.

    Returns:
        (y_true_sharon, y_pred_from_chen) for shared genes only.
    """
```

### 3. Create Calibration Report Notebook

Create `notebooks/crispr_screen_transfer/phase0_metric_calibration.py` as a Marimo notebook.

**Structure:**

```
Section 1: Load Chen 2019 and Sharon 2019 screen scores
Section 2: Compute positive and negative controls
Section 3: DRF for each metric (table + bar chart)
Section 4: Cross-screen ceiling (Chen vs Sharon correlation)
Section 5: Interpretation and go/no-go decision
```

**Section 3 output** — produce a table like:

| Metric | DRF | Positive ctrl score | Negative ctrl score | Verdict |
|--------|-----|---------------------|---------------------|---------|
| Spearman ρ | 0.XX | 0.XX | 0.00 | PASS/FAIL |
| Pearson r | 0.XX | 0.XX | 0.00 | PASS/FAIL |
| AUROC (sensitizer) | 0.XX | 0.XX | 0.50 | PASS/FAIL |
| Precision@50 | 0.XX | 0.XX | 0.09 | PASS/FAIL |
| Precision@100 | 0.XX | 0.XX | 0.09 | PASS/FAIL |
| Precision@200 | 0.XX | 0.XX | 0.09 | PASS/FAIL |
| Precision@500 | 0.XX | 0.XX | 0.09 | PASS/FAIL |
| neg-RMSE | 0.XX | 0.XX | 0.XX | PASS/FAIL |

**Verdict rule:** DRF >= 0.1 = PASS (metric can detect signal). DRF < 0.1 = FAIL (drop from primary analysis, report as secondary only).

**Section 4 output** — scatter plot: Chen score vs Sharon score for 17,091 shared genes, colored by hit status. Report Pearson r and Spearman ρ. This is the empirical ceiling for Design B.

**Section 5** — write a markdown cell with:
- Which metrics pass DRF threshold
- What the cross-screen ceiling is (Spearman ρ Chen vs Sharon)
- Recommended primary metrics for Phases 2–4
- Any surprises (e.g. if AUROC fails DRF, that would be unexpected for a fitness screen)

### 4. Save Calibration Artifacts

Save to `notebooks/crispr_screen_transfer/artifacts/phase0/`:

```
calibration_report.json     # DRF values for all metrics, both screens
cross_screen_ceiling.csv    # Chen vs Sharon scores for shared genes
calibration_summary.csv     # One row per metric: drf, verdict, recommended
```

The `calibration_report.json` schema:
```json
{
  "schema_version": "1.0.0",
  "timestamp_utc": "...",
  "screens": {
    "chen2019": {
      "n_genes": 19109,
      "n_sensitizers": ...,
      "n_resistors": ...,
      "metrics": {
        "spearman": {"drf": 0.XX, "positive_ctrl": 0.XX, "negative_ctrl": 0.0},
        ...
      }
    },
    "sharon2019": { ... }
  },
  "cross_screen_ceiling": {
    "n_shared_genes": 17091,
    "pearson_chen_sharon": 0.XX,
    "spearman_chen_sharon": 0.XX
  },
  "recommended_primary_metrics": ["spearman", "auroc_sensitizer", "precision_at_50", "precision_at_100"]
}
```

---

## Implementation Notes

### Fitness screens vs Perturb-seq: key difference

In Perturb-seq, each perturbation affects only ~5–50 of 19,000 genes transcriptomically. The "mean baseline" looks good because 99.9% of gene-perturbation pairs are near zero. In a fitness screen, every gene has a meaningful score (even if near zero, it's a real measurement). This means:

- MSE and Pearson are likely to be better calibrated here than in Perturb-seq
- The DRF values should be higher (closer to 1.0) for fitness screens
- If DRF is unexpectedly low, it suggests the screen has high technical noise

### Cross-screen ceiling interpretation

The Chen vs Sharon Spearman ρ is the **biological ceiling** for Design B. If ρ(Chen, Sharon) = 0.45, then no model trained on Chen data can be expected to predict Sharon data with ρ > 0.45. This ceiling should be reported prominently in all Design B results.

### Metric recommendations (expected outcome)

Based on the biology, we expect:
- Spearman ρ: DRF ~ 0.4–0.7 (PASS) — rank correlation is robust for fitness screens
- Pearson r: DRF ~ 0.3–0.6 (PASS) — linear correlation meaningful here
- AUROC: DRF ~ 0.3–0.6 (PASS) — hit classification is well-defined
- Precision@K: DRF ~ 0.2–0.5 (PASS) — depends on hit prevalence
- neg-RMSE: DRF ~ 0.1–0.3 (borderline) — may be diluted by near-zero genes

If any metric fails (DRF < 0.1), flag it and exclude from primary analysis in Phases 2–4.

---

## File Locations

| File | Action |
|------|--------|
| `src/crispr_al/metrics.py` | ADD `compute_drf`, `make_negative_control`, `make_positive_control_cross_screen`, `compute_calibration_report_with_hits` |
| `notebooks/crispr_screen_transfer/phase0_metric_calibration.py` | CREATE (Marimo notebook) |
| `notebooks/crispr_screen_transfer/artifacts/phase0/calibration_report.json` | OUTPUT |
| `notebooks/crispr_screen_transfer/artifacts/phase0/calibration_summary.csv` | OUTPUT |
| `notebooks/crispr_screen_transfer/artifacts/phase0/cross_screen_ceiling.csv` | OUTPUT |

---

## Tests

Add to `tests/test_metrics.py` (create if not exists):

```python
def test_compute_drf_perfect_predictor():
    """DRF should be 1.0 when positive control = ground truth."""

def test_compute_drf_null_predictor():
    """DRF should be 0.0 when positive control = negative control."""

def test_compute_drf_spearman_fitness_screen():
    """DRF for Spearman on synthetic fitness screen data should be > 0.1."""

def test_make_negative_control():
    """Negative control should be constant array equal to mean."""

def test_make_positive_control_cross_screen():
    """Cross-screen positive control should return aligned arrays for shared genes."""

def test_calibration_report_keys():
    """calibration_report_with_hits should return all expected metric keys."""
```

---

## Definition of Done

- [ ] `compute_drf()` implemented and tested in `src/crispr_al/metrics.py`
- [ ] `compute_calibration_report_with_hits()` implemented
- [ ] `make_negative_control()` and `make_positive_control_cross_screen()` implemented
- [ ] Marimo notebook `phase0_metric_calibration.py` runs end-to-end without errors
- [ ] `calibration_report.json` saved with DRF values for all 8 metrics on both screens
- [ ] `cross_screen_ceiling.csv` saved with Chen vs Sharon scores for 17,091 shared genes
- [ ] `calibration_summary.csv` saved with PASS/FAIL verdict per metric
- [ ] All new tests pass: `pytest tests/test_metrics.py`
- [ ] Notebook markdown cell in Section 5 states which metrics are recommended for Phases 2–4

---

## Dependency on Other Phases

- **Blocks:** Phase 2 (Design A/B), Phase 3 (Aim 2), Phase 4 (Active Learning) — all must use only metrics that pass DRF threshold
- **Depends on:** Nothing — this is the first phase
- **Handoff to Phase 2:** The `calibration_summary.csv` file lists recommended primary metrics. Phase 2 must use these metrics as primary and report others as secondary.
