"""
Recommendation Predictor — 10-Model GRU4Rec+UserFeat+BlendNet Ensemble
======================================================================
Reproduces the B榜第二轮 best score.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import Counter
import json
import sys
import argparse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCRIPT_DIR = Path(__file__).resolve().parent

# ── Constants ──
ITEM_FEAT_VOCAB = {'i_cat_01': 5, 'i_cat_02': 20, 'i_cat_03': 188, 'i_bucket_01': 11}
USER_FEAT_VOCAB = {'u_cat_01': 77, 'u_cat_02': 3, 'u_cat_03': 32, 'u_cat_04': 8,
                   'u_cat_05': 3, 'u_cat_06': 6, 'u_cat_07': 3, 'u_cat_08': 11}
NUM_ITEMS = 14065

# ── Best post-processing params (from stage3_result.json) ──
BUCKET_PARAMS = [
    {'cw': 5.0, 'rb': 0.3, 'pm': 0.1},   # Bucket 0: empty seq
    {'cw': 3.0, 'rb': 0.3, 'pm': 0.2},   # Bucket 1: len=1
    {'cw': 5.0, 'rb': 0.3, 'pm': 0.1},   # Bucket 2: len=2-3
    {'cw': 1.0, 'rb': 0.3, 'pm': 0.05},  # Bucket 3: len>=4
]

# Model dirs with trained checkpoints
MODEL_DIRS = [
    "workspace/outputs/recommendation/seed42_20260722_183719",
    "workspace/outputs/recommendation/seed123_20260722_183935",
    "workspace/outputs/recommendation/seed101_20260722_184446",
    "workspace/outputs/recommendation/seed303_20260722_184655",
    "workspace/outputs/recommendation/seed456_20260722_184101",
    "workspace/outputs/recommendation/seed789_20260722_184255",
    "workspace/outputs/recommendation/seed202_20260722_184542",
    "workspace/outputs/recommendation/seed404_20260722_185725",
    "workspace/outputs/recommendation/seed505_20260722_185900",
    "workspace/outputs/recommendation/seed606_20260722_190010",
]


def parse_seq(seq_str):
    if pd.isna(seq_str) or seq_str == '':
        return []
    return [int(x[1:]) for x in str(seq_str).split(',') if x]


# ═══════════════════════ Model Architecture ═══════════════════════

class GRU4Rec(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=128, num_layers=1, dropout=0.5, item_feat_emb_dim=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.item_id_embed = nn.Embedding(NUM_ITEMS + 1, embed_dim, padding_idx=0)
        self.item_feat_embeds = nn.ModuleDict({
            k: nn.Embedding(v, item_feat_emb_dim, padding_idx=0)
            for k, v in ITEM_FEAT_VOCAB.items()
        })
        total_feat_dim = len(ITEM_FEAT_VOCAB) * item_feat_emb_dim
        self.feat_proj = nn.Linear(embed_dim + total_feat_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers=num_layers,
                          batch_first=True, dropout=0 if num_layers == 1 else dropout)

    def _build_item_feat_table(self, item_df):
        item_df = item_df.copy()
        item_df['iid_idx'] = item_df['iid'].apply(lambda x: int(x[1:]))
        item_df = item_df.set_index('iid_idx')
        feat_ids = []
        for feat_name in ITEM_FEAT_VOCAB.keys():
            vals = np.zeros(NUM_ITEMS + 1, dtype=np.int64)
            for idx in item_df.index:
                if 1 <= idx <= NUM_ITEMS:
                    vals[idx] = item_df.loc[idx, feat_name]
            feat_ids.append(torch.tensor(vals, dtype=torch.long))
        self.register_buffer('item_feat_ids', torch.stack(feat_ids, dim=0))

    def _get_item_emb(self, item_ids):
        emb = self.item_id_embed(item_ids)
        if hasattr(self, 'item_feat_ids'):
            feats = []
            for k_idx, feat_name in enumerate(ITEM_FEAT_VOCAB.keys()):
                feat_vals = self.item_feat_ids[k_idx][item_ids]
                f_emb = self.item_feat_embeds[feat_name](feat_vals)
                feats.append(f_emb)
            all_feats = torch.cat(feats, dim=-1)
            emb = torch.cat([emb, all_feats], dim=-1)
            emb = self.feat_proj(emb)
        return emb

    def forward(self, item_seq):
        x = self._get_item_emb(item_seq)
        x = self.dropout(x)
        gru_out, _ = self.gru(x)
        return gru_out[:, -1, :]


class UserFeatModel(nn.Module):
    def __init__(self, user_embed_dim=16, hidden_dims=None, dropout=0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128]
        self.user_embeds = nn.ModuleDict({
            k: nn.Embedding(v, user_embed_dim, padding_idx=0)
            for k, v in USER_FEAT_VOCAB.items()
        })
        user_total_dim = len(USER_FEAT_VOCAB) * user_embed_dim
        layers = []
        in_dim = user_total_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h
        self.mlp = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, user_feats):
        B = user_feats.shape[0]
        embs = []
        for k_idx, k in enumerate(USER_FEAT_VOCAB.keys()):
            vals = user_feats[:, k_idx]
            embs.append(self.user_embeds[k](vals))
        x = torch.cat(embs, dim=-1)
        return self.mlp(x)


class BlendNet(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1), nn.Sigmoid()
        )

    def forward(self, seq_len_norm):
        return self.net(seq_len_norm.view(-1, 1)).squeeze(-1)


class JointModel(nn.Module):
    def __init__(self, gru, user_model):
        super().__init__()
        self.gru = gru
        self.user_model = user_model
        self.blend_net = BlendNet(hidden_dim=32)
        self.output = nn.Linear(gru.hidden_dim, NUM_ITEMS + 1)
        self.uf_proj = nn.Linear(user_model.output_dim, gru.hidden_dim)

    def forward(self, item_seq, user_feats, seq_len_norm):
        gru_hidden = self.gru(item_seq)
        gru_logits = self.output(gru_hidden)
        uf_hidden = self.user_model(user_feats)
        uf_hidden = self.uf_proj(uf_hidden)
        user_logits = F.linear(uf_hidden, self.output.weight, self.output.bias)
        alpha = self.blend_net(seq_len_norm).view(-1, 1)
        return alpha * gru_logits + (1 - alpha) * user_logits


class InferDataset(Dataset):
    def __init__(self, df, user_df, seq_col='dedup', seq_length=50):
        self.df = df.reset_index(drop=True)
        self.user_df = user_df.set_index('uid')
        self.seq_length = seq_length
        col = f'item_seq_{seq_col}'
        self.all_seqs = [parse_seq(s) for s in df[col]]
        self.uids = df['uid'].tolist()
        self.user_feat_cols = [f'u_cat_0{i}' for i in range(1, 9)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        raw_seq = self.all_seqs[idx]
        seq = raw_seq[-self.seq_length:] if len(raw_seq) > self.seq_length else raw_seq
        seq_len = len(seq)
        if seq_len < self.seq_length:
            padded = [0] * (self.seq_length - seq_len) + seq
        else:
            padded = seq
        seq_len_norm = min(seq_len, self.seq_length) / self.seq_length
        user_row = self.user_df.loc[uid]
        u_feats = torch.tensor([int(user_row[c]) for c in self.user_feat_cols], dtype=torch.long)
        return {
            'item_seq': torch.tensor(padded, dtype=torch.long),
            'user_feats': u_feats,
            'seq_len_norm': torch.tensor(seq_len_norm, dtype=torch.float32),
            'uid': uid,
        }


# ═══════════════════════ Inference ═══════════════════════

def load_model(ckpt_path, item_df):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    cfg = ckpt['config']
    gru = GRU4Rec(embed_dim=cfg['embed_dim'], hidden_dim=cfg['hidden_dim'],
                  num_layers=cfg['num_layers'], dropout=cfg['dropout'], item_feat_emb_dim=8)
    gru._build_item_feat_table(item_df)
    uf = UserFeatModel(user_embed_dim=16,
                       hidden_dims=[cfg['hidden_dim'], cfg['hidden_dim']],
                       dropout=cfg['dropout'])
    model = JointModel(gru, uf).to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, cfg


@torch.no_grad()
def compute_ensemble_logits(models, loader, uid_to_seq=None):
    all_fused, all_uids = [], []
    for batch in loader:
        item_seq = batch['item_seq'].to(DEVICE)
        user_feats = batch['user_feats'].to(DEVICE)
        seq_len_norm = batch['seq_len_norm'].to(DEVICE)
        fused_sum = None
        for model in models:
            fused = model(item_seq, user_feats, seq_len_norm)
            if fused_sum is None:
                fused_sum = fused
            else:
                fused_sum += fused
        fused_avg = fused_sum / len(models)
        all_fused.append(fused_avg.cpu().numpy())
        all_uids.extend(batch['uid'])
    all_fused = np.concatenate(all_fused, axis=0)
    all_raw_seqs = [uid_to_seq[uid] for uid in all_uids] if uid_to_seq else [[] for _ in all_uids]
    return all_fused, all_raw_seqs, all_uids


# ═══════════════════════ Post-processing ═══════════════════════

def build_covisitation(train_df, n_positions=3, decay=0.7, min_count=5):
    print(f"Building co-visitation (n_positions={n_positions}, min_count={min_count})...")
    co_vis = []
    for pos in range(n_positions):
        pair_counts, src_counts = {}, Counter()
        for _, row in train_df.iterrows():
            seq = parse_seq(row['item_seq_raw'])
            target = int(row['target_iid'][1:])
            if len(seq) > pos:
                src = seq[-(pos + 1)]
                pair_counts[(src, target)] = pair_counts.get((src, target), 0) + 1
                src_counts[src] += 1
        matrix = np.zeros((NUM_ITEMS + 1, NUM_ITEMS + 1), dtype=np.float32)
        for (src, tgt), cnt in pair_counts.items():
            if src_counts[src] >= min_count and 1 <= src <= NUM_ITEMS and 1 <= tgt <= NUM_ITEMS:
                matrix[src, tgt] = cnt / src_counts[src]
        co_vis.append(matrix)
    return co_vis


def apply_covisitation(logits, raw_seqs, co_vis, cw, n_positions=3, decay=0.7):
    boosted = logits.copy()
    for i in range(len(raw_seqs)):
        seq = raw_seqs[i]
        boost_vec = np.zeros(NUM_ITEMS + 1, dtype=np.float32)
        for p in range(min(n_positions, len(seq))):
            s = seq[-(p + 1)]
            if 0 < s <= NUM_ITEMS:
                boost_vec += (decay ** p) * co_vis[p][s]
        boosted[i] += cw * boost_vec
    return boosted


def apply_repurchase_boost(logits, raw_seqs, rb):
    boosted = logits.copy()
    for i in range(len(raw_seqs)):
        seq = raw_seqs[i]
        if len(seq) == 0:
            continue
        item_counts = Counter(seq)
        total = sum(item_counts.values())
        for item, cnt in item_counts.items():
            if 1 <= item <= NUM_ITEMS:
                boosted[i, item] += rb * (cnt / total)
    return boosted


def apply_pop_match(logits, target_probs, pm_alpha):
    adjusted = logits.copy()
    log_p = np.log(target_probs + 1e-10)
    adjusted[:, 1:] -= pm_alpha * log_p[1:]
    return adjusted


def compute_topk_predictions(logits, k=10):
    logits_no_pad = logits[:, 1:]
    topk_idx = np.argpartition(-logits_no_pad, k, axis=1)[:, :k]
    topk_vals = np.take_along_axis(logits_no_pad, topk_idx, axis=1)
    rerank = np.argsort(-topk_vals, axis=1)
    topk_idx = np.take_along_axis(topk_idx, rerank, axis=1)
    return topk_idx + 1


def get_bucket_indices(truncated_seqs, n_buckets=4):
    buckets = [[] for _ in range(n_buckets)]
    for i, sl in enumerate(truncated_seqs):
        if sl == 0:
            buckets[0].append(i)
        elif sl == 1:
            buckets[1].append(i)
        elif sl <= 3:
            buckets[2].append(i)
        else:
            buckets[3].append(i)
    return buckets


def main():
    parser = argparse.ArgumentParser(description="Recommendation Predictor")
    parser.add_argument("--data", default="workspace/data/rec", help="Path to data directory")
    parser.add_argument("--output", default="workspace/outputs/recommendation/B2.csv", help="Output CSV path")
    parser.add_argument("--model-dir", default="workspace/outputs/recommendation",
                       help="Directory containing seed* model checkpoints")
    args = parser.parse_args()

    DATA_DIR = Path(args.data)
    MODEL_DIR = Path(args.model_dir)
    OUT_PATH = Path(args.output)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Data: {DATA_DIR}")
    print(f"Models: {MODEL_DIR}")
    print(f"Output: {OUT_PATH}")
    print(f"Device: {DEVICE}")

    # ── Load data ──
    print("\nLoading data...")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    user_df = pd.read_csv(DATA_DIR / "user.csv")
    item_df = pd.read_csv(DATA_DIR / "item.csv")

    # ── Split holdout using saved holdout_uids (same as training time) ──
    holdout_path = Path("workspace/outputs/recommendation/seed42_20260722_171809/holdout_uids.npy")
    holdout_uids = set(np.load(holdout_path).tolist())
    train_uids = set(train_df['uid'].unique()) - holdout_uids
    train_sub = train_df[train_df['uid'].isin(train_uids)].reset_index(drop=True)
    print(f"Train sub: {len(train_sub)}, Holdout: {len(holdout_uids)}")

    # ── Build co-visitation from train_sub only ──
    co_vis = build_covisitation(train_sub, n_positions=3, decay=0.7, min_count=5)

    # ── Target probabilities for Pop Match (from train_sub) ──
    target_counts = Counter()
    for _, row in train_sub.iterrows():
        t = int(row['target_iid'][1:])
        target_counts[t] += 1
    target_probs = np.zeros(NUM_ITEMS + 1, dtype=np.float32)
    total = sum(target_counts.values())
    for t, c in target_counts.items():
        if 1 <= t <= NUM_ITEMS:
            target_probs[t] = c / total
    target_probs[target_probs == 0] = 1.0 / (total + NUM_ITEMS)

    # ── Load models ──
    print("\nLoading models...")
    models = []
    for md in MODEL_DIRS:
        ckpt_path = Path(md) / "best_model.pt"
        if ckpt_path.exists():
            model, cfg = load_model(ckpt_path, item_df)
            models.append(model)
            print(f"  Loaded: {Path(md).name} (val_ndcg={cfg.get('best_val_ndcg', '?')})")
        else:
            print(f"  MISSING: {ckpt_path}")
    print(f"Total: {len(models)} models")

    if len(models) == 0:
        print("ERROR: No models loaded!")
        return

    # ── Test set inference ──
    print("\nTest set inference...")
    test_ds = InferDataset(test_df, user_df, seq_col='dedup', seq_length=50)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)

    test_seq_map = {}
    for _, row in test_df.iterrows():
        test_seq_map[row['uid']] = parse_seq(row['item_seq_raw'])

    test_logits, test_raw_seqs, test_uids = compute_ensemble_logits(models, test_loader, test_seq_map)

    # ── Bucketed post-processing (hardcoded best params) ──
    print("\nApplying bucketed post-processing...")
    test_trunc_seqs = []
    for uid in test_uids:
        seq = test_seq_map[uid]
        test_trunc_seqs.append(min(len(seq), 50))

    test_buckets = get_bucket_indices(test_trunc_seqs, n_buckets=4)
    for b_idx, b_indices in enumerate(test_buckets):
        bp = BUCKET_PARAMS[b_idx]
        print(f"  Bucket {b_idx}: n={len(b_indices)}, cw={bp['cw']}, rb={bp['rb']}, pm={bp['pm']}")

    final_logits = test_logits.copy()
    for b_idx, b_indices in enumerate(test_buckets):
        if len(b_indices) == 0:
            continue
        cw, rb, pm = BUCKET_PARAMS[b_idx]['cw'], BUCKET_PARAMS[b_idx]['rb'], BUCKET_PARAMS[b_idx]['pm']
        b_logits = final_logits[b_indices]
        b_seqs = [test_raw_seqs[i] for i in b_indices]
        if cw > 0:
            b_logits = apply_covisitation(b_logits, b_seqs, co_vis, cw)
        if rb > 0:
            b_logits = apply_repurchase_boost(b_logits, b_seqs, rb)
        if pm > 0:
            b_logits = apply_pop_match(b_logits, target_probs, pm)
        final_logits[b_indices] = b_logits

    # ── Generate predictions ──
    topk = compute_topk_predictions(final_logits, k=10)
    predictions = []
    for i, uid in enumerate(test_uids):
        pred_items = [f"i{idx:06d}" for idx in topk[i]]
        predictions.append(','.join(pred_items))

    sub_df = pd.DataFrame({'uid': test_uids, 'prediction': predictions})
    sub_df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved: {OUT_PATH} ({len(sub_df)} rows)")

    # Quick validation
    assert len(sub_df) == 10000, f"Expected 10000 rows, got {len(sub_df)}"
    for pred in predictions:
        items = pred.split(',')
        assert len(items) == 10, f"Expected 10 items, got {len(items)}"
        assert len(set(items)) == 10, f"Duplicate items in prediction"
    print("Format validation: OK")


if __name__ == "__main__":
    main()
