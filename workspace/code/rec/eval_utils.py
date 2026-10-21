"""推荐任务评估工具 — 纯函数，不含模型/超参/训练逻辑。

Agent 训练和推理脚本直接 import 使用，确保评估实现一致、避免重复造轮子。

用法:
    训练脚本:
        from eval_utils import compute_ndcg_at_k_torch, compute_hr_at_k_torch, make_trunc_mask

    推理脚本:
        from eval_utils import compute_ndcg_at_k, compute_hr_at_k, make_trunc_mask
"""

import numpy as np
import torch


# ═══════════════════════════════════════════════════════════════
# NDCG / HR — NumPy 版（推理/后处理用）
# ═══════════════════════════════════════════════════════════════

def compute_ndcg_at_k(logits, targets, k=10):
    """计算 NDCG@k (NumPy 版)。

    Args:
        logits: (n, num_items) 或 (n, num_items+1) — 如果包含 padding 位(index 0)，
                函数会自动检测并去掉。
        targets: (n,) — 1-indexed 的 target item ID
        k: top-k

    Returns:
        float: 平均 NDCG@k

    约定：始终传含 padding 的 logits (n, num_items+1) + 1-indexed targets。
    如果已提前去掉 padding，负责将 targets 转为 0-indexed 对齐。
    """
    need_offset = False
    if logits.shape[1] == int(targets.max()) + 1:
        logits = logits[:, 1:]   # 去掉 padding index 0
        need_offset = True       # topk 是 0-indexed，需 +1 对齐 1-indexed targets

    # Top-k indices
    topk_idx = np.argpartition(-logits, k, axis=1)[:, :k]
    # Rerank by score descending
    topk_vals = np.take_along_axis(logits, topk_idx, axis=1)
    rerank = np.argsort(-topk_vals, axis=1)
    topk_idx = np.take_along_axis(topk_idx, rerank, axis=1)

    # 如果 targets 是 1-indexed 但 topk 是 0-indexed，需要 +1 对齐
    if need_offset:
        topk_idx = topk_idx + 1

    # Hit detection
    targets_2d = targets.reshape(-1, 1).astype(np.int64)
    hits = (topk_idx == targets_2d).astype(np.float32)

    # DCG: 1 / log2(rank + 1)
    positions = np.arange(1, k + 1, dtype=np.float32)
    dcg = (hits / np.log2(positions + 1)).sum(axis=1)

    # IDCG: target at rank=1
    idcg = 1.0 / np.log2(2.0)

    return float((dcg / idcg).mean())


def compute_hr_at_k(logits, targets, k=10):
    """计算 HitRate@k (NumPy 版)。"""
    need_offset = False
    if logits.shape[1] == int(targets.max()) + 1:
        logits = logits[:, 1:]
        need_offset = True

    topk_idx = np.argpartition(-logits, k, axis=1)[:, :k]
    topk_vals = np.take_along_axis(logits, topk_idx, axis=1)
    rerank = np.argsort(-topk_vals, axis=1)
    topk_idx = np.take_along_axis(topk_idx, rerank, axis=1)

    if need_offset:
        topk_idx = topk_idx + 1

    targets_2d = targets.reshape(-1, 1).astype(np.int64)
    hits = (topk_idx == targets_2d).any(axis=1).astype(np.float32)
    return float(hits.mean())


# ═══════════════════════════════════════════════════════════════
# NDCG / HR — PyTorch 版（训练时验证用）
# ═══════════════════════════════════════════════════════════════

def compute_ndcg_at_k_torch(logits, targets, k=10):
    """计算 NDCG@k (PyTorch 版，no_grad 内调用)。

    Args:
        logits: (n, num_items) 或 (n, num_items+1)
        targets: (n,) — 1-indexed
        k: top-k

    Returns:
        float
    """
    with torch.no_grad():
        need_offset = False
        if logits.shape[1] == targets.max().item() + 1:
            logits = logits[:, 1:]
            need_offset = True

        _, topk_idx = torch.topk(logits, k, dim=1)

        if need_offset:
            topk_idx = topk_idx + 1

        targets_2d = targets.view(-1, 1)
        hits = (topk_idx == targets_2d).float()

        positions = torch.arange(1, k + 1, device=logits.device, dtype=torch.float32)
        dcg = (hits / torch.log2(positions + 1)).sum(dim=1)

        idcg = 1.0 / torch.log2(torch.tensor(2.0, device=logits.device))

        return (dcg / idcg).mean().item()


def compute_hr_at_k_torch(logits, targets, k=10):
    """计算 HitRate@k (PyTorch 版)。"""
    with torch.no_grad():
        need_offset = False
        if logits.shape[1] == targets.max().item() + 1:
            logits = logits[:, 1:]
            need_offset = True

        _, topk_idx = torch.topk(logits, k, dim=1)

        if need_offset:
            topk_idx = topk_idx + 1

        targets_2d = targets.view(-1, 1)
        hits = (topk_idx == targets_2d).any(dim=1).float()
        return hits.mean().item()


# ═══════════════════════════════════════════════════════════════
# 多级截断 Mask（验证集 / holdout 评估用）
# ═══════════════════════════════════════════════════════════════

def make_trunc_mask(n_samples, seed, dist=None):
    """生成多级截断 mask，模拟测试集序列长度分布。

    对数据集中每个样本分配一个截断级别：
      0 = 完全清空（模拟冷启动）
      1 = 只保留最后 1 个产品
      2 = 保留最后 2-3 个产品
      3 = 保留完整序列

    Args:
        n_samples: 样本数
        seed: 随机种子（固定以确保跨 epoch 一致）
        dist: 四级分布概率，默认 (0.35, 0.10, 0.45, 0.10)
              即 35% 空 / 10% len=1 / 45% len=2-3 / 10% 完整

    Returns:
        list[int]: 长度 n_samples，每个元素 ∈ {0, 1, 2, 3}

    用法:
        # 验证时：在 Dataset.__init__ 中生成，确保跨 epoch 固定
        self.trunc_mask = make_trunc_mask(len(data), seed=42)

        # 在 __getitem__ 中：
        level = self.trunc_mask[idx]
        if level == 0:      seq = []
        elif level == 1:    seq = raw_seq[-1:]
        elif level == 2:    keep = random.randint(2, min(3, len(raw_seq)))
                            seq = raw_seq[-keep:]
        else:               seq = raw_seq[:]
    """
    if dist is None:
        dist = (0.35, 0.10, 0.45, 0.10)

    p_empty, p_len1, p_len23, p_full = dist
    assert abs(p_empty + p_len1 + p_len23 + p_full - 1.0) < 0.001, \
        f"分布之和必须为 1.0，当前: {p_empty}+{p_len1}+{p_len23}+{p_full}"

    rng = np.random.RandomState(seed)
    mask = []
    for _ in range(n_samples):
        r = rng.random()
        if r < p_empty:
            mask.append(0)
        elif r < p_empty + p_len1:
            mask.append(1)
        elif r < p_empty + p_len1 + p_len23:
            mask.append(2)
        else:
            mask.append(3)

    # 打印分布统计
    counts = {i: mask.count(i) for i in range(4)}
    print(f"截断分布 (seed={seed}): "
          f"空={counts[0]}({counts[0]/n_samples:.2f}) "
          f"len1={counts[1]}({counts[1]/n_samples:.2f}) "
          f"len2-3={counts[2]}({counts[2]/n_samples:.2f}) "
          f"完整={counts[3]}({counts[3]/n_samples:.2f})")

    return mask
