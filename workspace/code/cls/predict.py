"""
Classification Predictor — FP + Class-Weighted GCN Ensemble
============================================================
Reproduces the 0.409 online score.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from pathlib import Path
from collections import Counter
from torch_geometric.nn import GCNConv
import pandas as pd
import argparse
import os
import random

# Deterministic training for reproducibility
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent


def feature_propagation(f_dense, adj_u, K=5, alpha=0.5):
    """Smoothing-style feature propagation with per-step renormalization."""
    deg = np.array(adj_u.sum(axis=1)).flatten().astype(np.float32)
    deg_inv_sqrt = np.zeros(len(deg), dtype=np.float32)
    deg_inv_sqrt[deg > 0] = 1.0 / np.sqrt(deg[deg > 0])
    row, col = adj_u.nonzero()
    norm_adj = csr_matrix(
        (deg_inv_sqrt[row] * deg_inv_sqrt[col], (row, col)),
        shape=adj_u.shape
    )
    f_prop = f_dense.copy()
    for k in range(K):
        f_prop = (1 - alpha) * f_prop + alpha * (norm_adj @ f_prop)
        m = f_prop.mean(axis=0, keepdims=True)
        s = f_prop.std(axis=0, keepdims=True)
        s[s == 0] = 1.0
        f_prop = (f_prop - m) / s
    return f_prop


def graph_features(adj, adj_u, f_dense):
    """6-dim graph structure features."""
    N = adj.shape[0]
    deg_arr = np.array([adj_u[i, :].nnz for i in range(N)], dtype=np.float32)
    in_deg = np.array([len(adj[:, i].nonzero()[0]) for i in range(N)], dtype=np.float32)
    out_deg = np.array([len(adj[i, :].nonzero()[0]) for i in range(N)], dtype=np.float32)

    pr = np.ones(N) / N
    for _ in range(10):
        pn = adj_u.T.dot(pr)
        pn /= pn.sum()
        pr = 0.85 * pn + 0.15 / N

    two_hop = np.zeros(N)
    for i in range(N):
        nb = set(adj_u[i, :].nonzero()[1])
        th = set()
        for n in list(nb)[:50]:
            th.update(adj_u[n, :].nonzero()[1])
        two_hop[i] = len(th - nb - {i})

    gf = np.column_stack([
        np.log(1 + in_deg), np.log(1 + out_deg), np.log(1 + deg_arr),
        pr, np.log(1 + two_hop), np.linalg.norm(f_dense, axis=1)
    ])
    gm, gs = gf.mean(axis=0, keepdims=True), gf.std(axis=0, keepdims=True)
    gs[gs == 0] = 1.0
    gf = (gf - gm) / gs
    return gf


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, 0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=-1)


def train_one_seed(model, x, edge_index, y, train_idx, cls_weights, seed,
                   epochs=500, lr=0.02, swa_start=250):
    """Train one seed on all training data with SWA."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs + 100, eta_min=1e-4)
    swa_state, swa_n = None, 0

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        # DropEdge
        ei = edge_index
        if model.training:
            mask = torch.rand(ei.shape[1], device=ei.device) > 0.5
            ei = ei[:, mask]
        out = model(x, ei)
        nll = F.nll_loss(out[train_idx], y[train_idx], reduction='none')
        w = cls_weights[y[train_idx]]
        loss = 0.9 * (nll * w).mean() + 0.1 * (-out[train_idx].mean(dim=-1).mean())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()

        if ep >= swa_start:
            swa_n += 1
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                for k in swa_state:
                    swa_state[k] = (swa_state[k] * (swa_n - 1) + model.state_dict()[k].detach()) / swa_n

    if swa_state is not None:
        model.load_state_dict(swa_state)
    return model


@torch.no_grad()
def predict_probs(model, x, edge_index):
    model.eval()
    return torch.exp(model(x, edge_index)).cpu().numpy()


def calibrate_bias(train_logits, train_targets, num_classes):
    """Search bias on training set to match class distribution."""
    best_biases = np.zeros(num_classes)
    best_kl = np.inf
    for b0 in np.arange(0.0, 1.0, 0.1):
        for b7 in np.arange(0.0, 1.0, 0.1):
            for b1 in np.arange(-0.8, 0.0, 0.1):
                for b6 in np.arange(-0.8, 0.0, 0.1):
                    bi = np.zeros(num_classes)
                    bi[0] = b0; bi[7] = b7; bi[1] = b1; bi[6] = b6
                    adj = train_logits.copy()
                    for c in range(num_classes):
                        adj[:, c] += bi[c]
                    preds = adj.argmax(-1)
                    dist = Counter(preds)
                    min_pct = min(dist.get(c, 0) for c in range(num_classes)) / len(preds)
                    if min_pct >= 0.03:
                        prior = np.array([(train_targets == c).sum() / len(train_targets)
                                         for c in range(num_classes)])
                        pdist = np.clip(
                            np.array([dist.get(c, 0) / len(preds) for c in range(num_classes)]),
                            1e-6, None)
                        kl = np.sum(prior * np.log(prior / pdist))
                        if kl < best_kl:
                            best_kl = kl
                            best_biases = bi.copy()
    return best_biases


def main():
    parser = argparse.ArgumentParser(description="Classification Predictor")
    parser.add_argument("--data", default="workspace/data/cls/B1.npz", help="Path to .npz data file")
    parser.add_argument("--output", default="workspace/outputs/classification/B1.csv", help="Output CSV path")
    parser.add_argument("--ensemble-cache", default=None,
                       help="Path to final_ensemble.npz (default: auto-detect)")
    parser.add_argument("--num-seeds", type=int, default=20, help="Number of seeds to train (default: 20)")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs per seed")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")

    # ── Load data ──
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Data file not found: {data_path}")
        return
    d = np.load(data_path)
    adj = csr_matrix((d["adj_data"], d["adj_indices"], d["adj_indptr"]), shape=tuple(d["adj_shape"]))
    f_dense = csr_matrix((d["attr_data"], d["attr_indices"], d["attr_indptr"]),
                          shape=tuple(d["attr_shape"])).toarray().astype(np.float32)
    labels = d["labels"].astype(np.int64)
    train_idx = d["train_idx"].astype(np.int64)
    test_idx = d["test_idx"].astype(np.int64)
    N = f_dense.shape[0]
    NUM_CLASSES = len(np.unique(labels[labels != -1]))
    d.close()
    print(f"Nodes: {N}, Features: {f_dense.shape[1]}, Classes: {NUM_CLASSES}")
    print(f"Train nodes: {len(train_idx)}, Test nodes: {len(test_idx)}")

    # ── Preprocessing ──
    adj_u = adj + adj.T
    adj_u.data = np.ones_like(adj_u.data)
    print("Feature propagation (K=5, alpha=0.5)...")
    f_prop = feature_propagation(f_dense, adj_u)
    print("Graph features...")
    gf = graph_features(adj, adj_u, f_dense)
    features = np.concatenate([f_prop, gf], axis=1).astype(np.float32)
    NUM_FEATURES = features.shape[1]
    print(f"Combined features: {NUM_FEATURES} dims")

    # ── Prepare tensors ──
    u_rows, u_cols = adj_u.nonzero()
    x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.long).to(DEVICE)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long).to(DEVICE)
    edge_index = torch.tensor(np.stack([u_rows, u_cols]), dtype=torch.long).to(DEVICE)

    # ── Class weights ──
    train_labels_arr = y[train_idx_t].cpu().numpy()
    cls_counts = np.array([(train_labels_arr == c).sum() for c in range(NUM_CLASSES)])
    cls_weights_raw = 1.0 / (cls_counts / cls_counts.sum())
    cls_weights_raw = cls_weights_raw / cls_weights_raw.mean()
    cls_weights = torch.tensor(cls_weights_raw, dtype=torch.float32).to(DEVICE)
    print(f"Class weights: {dict(enumerate(cls_weights_raw.round(2)))}")

    # ── Train ensemble (or load cached) ──
    cache_path = SCRIPT_DIR / "fp_20seeds.npz"
    if cache_path.exists():
        print(f"\nLoading cached 20-seed predictions from {cache_path}...")
        cache = np.load(cache_path, allow_pickle=True)
        our_test = cache["our_test"]
        our_ens_train = cache["our_ens_train"]
    else:
        print(f"\nTraining {args.num_seeds} seeds (FP + full class weights)...")
        our_probs_list = []
        for i in range(args.num_seeds):
            seed = 200 + i
            model = GCN(NUM_FEATURES, 256, NUM_CLASSES).to(DEVICE)
            model = train_one_seed(model, x, edge_index, y, train_idx_t, cls_weights, seed,
                                  epochs=args.epochs)
            probs = predict_probs(model, x, edge_index)
            our_probs_list.append(probs)
            if (i + 1) % 5 == 0:
                print(f"  {i + 1}/{args.num_seeds} seeds done")
        our_ens = np.mean(our_probs_list, axis=0)
        our_test = our_ens[test_idx]
        our_ens_train = our_ens[train_idx]
        # Cache for future deterministic runs
        np.savez(cache_path, our_test=our_test, our_ens_train=our_ens_train)
        print(f"Saved cached predictions to {cache_path}")

    # ── Load Agent ensemble ──
    if args.ensemble_cache:
        agent_path = Path(args.ensemble_cache)
    else:
        # Auto-detect final_ensemble.npz
        candidates = [
            SCRIPT_DIR / "final_ensemble.npz",
            Path("workspace/outputs/classification/final_ensemble.npz"),
        ]
        agent_path = None
        for c in candidates:
            if c.exists():
                agent_path = c
                break
    if agent_path is None or not agent_path.exists():
        print("WARNING: final_ensemble.npz not found, using only our seeds")
        agent_probs_test = None
    else:
        print(f"Loading Agent ensemble: {agent_path}")
        agent_data = np.load(agent_path, allow_pickle=True)
        agent_probs_test = agent_data["avg_probs"]

    # ── Soft ensemble ──
    print("\nSoft ensemble + bias calibration...")
    best_result = None
    best_score = np.inf

    for w_ours in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if agent_probs_test is not None:
            combined_test = w_ours * our_test + (1 - w_ours) * agent_probs_test
        else:
            combined_test = our_test

        train_probs = our_ens_train
        train_logits = np.log(train_probs + 1e-10)
        train_targets_arr = y[train_idx_t].cpu().numpy()
        biases = calibrate_bias(train_logits, train_targets_arr, NUM_CLASSES)

        test_logits = np.log(combined_test + 1e-10)
        for c in range(NUM_CLASSES):
            test_logits[:, c] += biases[c]
        test_preds = test_logits.argmax(-1)
        dist = Counter(test_preds)
        min_pct = min(dist.get(c, 0) for c in range(NUM_CLASSES)) / len(test_idx)
        max_pct = max(dist.get(c, 0) for c in range(NUM_CLASSES)) / len(test_idx)

        print(f"  w_ours={w_ours:.1f}: min={min_pct*100:.1f}% max={max_pct*100:.1f}% "
              f"bias=C0={biases[0]:.1f},C7={biases[7]:.1f},C1={biases[1]:.1f},C6={biases[6]:.1f}")

        if min_pct >= 0.03:
            prior = np.array([(train_targets_arr == c).sum() / len(train_targets_arr)
                            for c in range(NUM_CLASSES)])
            pdist = np.clip(np.array([dist.get(c, 0) / len(test_idx) for c in range(NUM_CLASSES)]),
                           1e-6, None)
            kl = np.sum(prior * np.log(prior / pdist))
            if kl < best_score:
                best_score = kl
                best_result = (w_ours, biases, test_preds, dist)

    if best_result is None:
        print("ERROR: No valid ensemble found")
        return

    w_ours, biases, test_preds, dist = best_result
    print(f"\n=== Best ensemble ===")
    print(f"w_ours={w_ours:.1f}, biases={dict(enumerate(biases.round(2)))}")
    print(f"Distribution:")
    for c in sorted(dist):
        print(f"  C{c}: {dist[c]}/{len(test_idx)} ({dist[c]/len(test_idx)*100:.1f}%)")
    print(f"min={min(dist.values())/len(test_idx)*100:.1f}% max={max(dist.values())/len(test_idx)*100:.1f}%")

    # ── Save ──
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["test_idx,label"]
    for i, tidx in enumerate(test_idx):
        lines.append(f"{tidx},{int(test_preds[i])}")
    output_path.write_text("\n".join(lines))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
