"""Classification task evaluation tools — unified validation framework that all training scripts must use.

Agent training scripts import and use this directly to ensure consistent evaluation methods and comparable scores, avoiding each script writing its own validation logic.

Usage:
    from eval_utils import split_train_val, compute_accuracy, evaluate_seeds, get_node_degrees

Core principles (Agent must follow):
    1. Stratified split: stratify by class + degree jointly, avoid validation set bias toward high-degree nodes
    2. Fixed seed: use the same split seed across experiments, otherwise scores are not comparable
    3. Validation set degree distribution must align with test set: high-degree nodes (>50) ~5%, not ~30% as in random splits
    4. Full-seed averaging: when comparing configs, use mean + std across all seeds; do not pick top-k seeds for ensembling
    5. Small val improvements are not real improvements: with limited validation nodes, small fluctuations are sampling noise, do not chase them
"""

import numpy as np
from sklearn.model_selection import train_test_split


def get_node_degrees(adj_csr):
    """Compute undirected node degrees of the graph (for degree-aware splitting).

    Args:
        adj_csr: scipy.sparse.csr_matrix — adjacency matrix (directed or undirected both work)

    Returns:
        degrees: (n,) float32 array — undirected degree of each node
    """
    adj_u = adj_csr + adj_csr.T
    adj_u.data = np.ones_like(adj_u.data)
    return np.array([adj_u[i, :].nnz for i in range(adj_csr.shape[0])], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# Train/Validation Split (Degree-Aware Version)
# ═══════════════════════════════════════════════════════════════

def split_train_val(train_idx, labels, val_ratio=0.2, seed=42, node_degrees=None):
    """Stratified train/validation split. If node_degrees is provided, use degree-aware splitting
    to ensure validation set degree distribution matches the test set (high-degree nodes ~5% instead of ~30%).

    Without degree awareness, random stratified splitting puts many high-degree nodes into the validation set
    (avg degree ~52), but the test set avg degree is only ~39 and degree >50 is only ~5%.
    This causes inflated local scores that drop sharply online.

    Args:
        train_idx: (n,) int array — training node indices
        labels: (total_nodes,) int array — all node labels
        val_ratio: proportion of training nodes reserved for validation
        seed: fixed random seed
        node_degrees: (total_nodes,) float array — node degrees. If None, fall back to pure label stratification

    Returns:
        train_sub_idx, val_idx
    """
    if node_degrees is None:
        train_labels = labels[train_idx]
        train_sub, val_sub = train_test_split(
            train_idx, test_size=val_ratio, stratify=train_labels, random_state=seed,
        )
        return train_sub, val_sub

    # Degree-aware split: create (label, degree_bin) joint stratification
    train_labels = labels[train_idx]
    train_degs = node_degrees[train_idx]

    # Degree bucketing: aligned with test set distribution. Test set degree >50 is ~5%
    degree_bins = [0, 20, 30, 40, 55, 10000]
    deg_bin_idx = np.digitize(train_degs, degree_bins) - 1

    # Joint stratification labels: "label_degBin"
    joint_stratify = np.array([
        f"{train_labels[i]}_{deg_bin_idx[i]}" for i in range(len(train_idx))
    ])

    # Handle classes with insufficient sample count (merge to adjacent bucket)
    from collections import Counter
    cnt = Counter(joint_stratify)
    for i, js in enumerate(joint_stratify):
        if cnt[js] < 2:
            # Merge to adjacent degree bucket
            parts = js.split("_")
            new_bin = max(0, int(parts[1]) - 1)
            joint_stratify[i] = f"{parts[0]}_{new_bin}"

    try:
        train_sub, val_sub = train_test_split(
            train_idx, test_size=val_ratio, stratify=joint_stratify, random_state=seed,
        )
    except ValueError:
        # Fall back to pure label stratification
        train_sub, val_sub = train_test_split(
            train_idx, test_size=val_ratio, stratify=train_labels, random_state=seed,
        )

    # Check validation set degree distribution; correct if too skewed — target: high-degree nodes (>50) < 12%
    val_degs = node_degrees[val_sub]
    high_deg_pct = (val_degs > 50).sum() / len(val_sub)
    if high_deg_pct > 0.12:
        high_val = val_sub[val_degs > 50]
        low_train = train_sub[node_degrees[train_sub] <= 35]
        n_swap = min(len(high_val), len(low_train))
        n_swap = int(n_swap * 0.85)  # Move 85% of high-degree nodes away
        if n_swap > 0:
            swap_out = np.random.RandomState(seed).choice(high_val, n_swap, replace=False)
            swap_in = np.random.RandomState(seed+1).choice(low_train, n_swap, replace=False)
            val_sub = np.array([v for v in val_sub if v not in swap_out] + list(swap_in))
            train_sub = np.array([t for t in train_sub if t not in swap_in] + list(swap_out))

    return train_sub, val_sub


# ═══════════════════════════════════════════════════════════════
# Accuracy Computation
# ═══════════════════════════════════════════════════════════════

def compute_accuracy(predictions, labels):
    """Compute classification accuracy.

    Args:
        predictions: (n,) int array — predicted class indices
        labels: (n,) int array — ground truth class indices

    Returns:
        float: accuracy (0.0 ~ 1.0)
    """
    return float((predictions == labels).mean())


def compute_per_class_accuracy(predictions, labels, num_classes):
    """Compute per-class accuracy, for analyzing class imbalance effects.

    Args:
        predictions: (n,) int array
        labels: (n,) int array
        num_classes: total number of classes

    Returns:
        dict: {class_idx: accuracy}, accuracy is 0.0 for missing classes
    """
    result = {}
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() > 0:
            result[c] = float((predictions[mask] == c).mean())
        else:
            result[c] = 0.0
    return result


# ═══════════════════════════════════════════════════════════════
# Ensemble Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_seeds(val_accs):
    """Evaluate multi-seed training results, returning mean + standard deviation.

    When comparing different configs, must use the mean across all seeds (not top-k mean).
    top-k selection = using validation set to pick seeds = overfitting to the validation set.

    Args:
        val_accs: list[float] — validation accuracy for each seed

    Returns:
        dict: {"mean": float, "std": float, "min": float, "max": float, "n": int}
    """
    arr = np.array(val_accs)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": len(val_accs),
    }


def ensemble_logits_average(all_logits):
    """Simple averaging of logits from multiple models (balanced ensemble, equal weights per seed).

    Args:
        all_logits: list of (n_test, n_classes) float arrays

    Returns:
        (n_test, n_classes) float array — averaged logits
    """
    return np.mean(np.stack(all_logits), axis=0)


# ═══════════════════════════════════════════════════════════════
# Validation Methodology (reference for Agent, not code)
# ═══════════════════════════════════════════════════════════════

VALIDATION_METHODOLOGY = """
## Local Validation Methodology Standard

### Splitting Method (Important! Must be degree-aware)
- **Must use degree-aware splitting**: pass node_degrees parameter when calling split_train_val
- First obtain degrees via get_node_degrees(adj_csr), then pass them into split_train_val
- Without degree awareness, random stratified splitting puts many high-degree nodes into validation set, inflating local scores
- Test set avg degree ~39, degree >50 only ~5%; validation set must match this distribution

### Bias Calibration
- **Search for bias on training set**, do not tune bias on validation set
- Validation set has only ~900 nodes, minority classes have only 40 samples — bias search is unreliable there
- Training set has 5000+ nodes, each class has at least 250+ samples
- After adding bias to training set logits, check distribution (each class >= 3%); do NOT check accuracy ON THE TRAINING SET (model overfits there)
- **CRITICAL: After bias calibration, verify on validation set that accuracy did NOT drop >0.02. If it dropped more, bias is too aggressive — reduce it. Bias should only nudge ambiguous cases, not override confident predictions.**
- **Distribution target is TRAINING distribution, NOT uniform**: if training has C1=22%, C6=25%, similar prediction proportions are EXPECTED. Only ensure minority classes aren't suppressed to 0%.

### Model Comparison
- Architecture screening phase: 1 seed per architecture is sufficient, compare single-seed val scores horizontally
- Hyperparameter tuning phase: multiple seeds per config, compare using means (not top-k means)
- Ensemble phase: all trained seeds participate in ensembling, do simple averaging. The only valid reason to exclude a seed is training failure (val_acc near random level, indicating a bug, not "lower than other seeds"). Do NOT filter by relative val score.

### Score Interpretation
- Degree-aware val will be 0.03-0.05 lower than the old version — this is normal and more accurate
- Small val improvements are likely sampling noise, do not chase them
- Do not use old non-degree-aware scores for comparison — old scores are inflated

### Red Lines
- Forbidden to call split_train_val without passing node_degrees
- Forbidden to pick top-k seeds by val score for ensembling (this is overfitting to validation set). Note: excluding seeds that clearly failed to train (val_acc ≈ random) is NOT "picking top-k" — it's removing buggy models. The difference: "failed" = absolute (near random), "lower quality" = relative (worse than other seeds)
- Forbidden to tune bias on validation set — too few samples, unreliable; tune on training set instead
- Local val goes up but online goes down → you are overfitting to validation set, stop fine-tuning immediately
"""
