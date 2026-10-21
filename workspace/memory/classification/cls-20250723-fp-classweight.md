# 分类任务实验记忆 — FP+类权重集成

## 日期

2025-07-23

## 关键突破（相比v3）

### v3回顾

- GCN 53种子集成, K=5 特征传播(APPNP式), lr=0.02最优
- 偏置校准: {0:+0.4, 1:-0.4, 6:-0.6, 7:+0.4}
- val=0.4639

### 本次突破

1. **完全逆频率类别权重**: `w_c = (N/N_c) / mean(N/N_c)` — **最关键突破**
   - C0 weight=1.8x, C7 weight=2.13x, C1=0.4x, C6=0.36x
   - 效果: min class 从不加权重的 2.5% 直接跃升到 6.1%(无需任何偏置校准!)
   - 之前 sqrt 逆频率(min=3.5%)远不如完全逆频率(6.1%)
2. **全数据训练**: 使用全部 6120 训练节点, 无验证集划分, 纯 SWA(epoch 250-500)
   - 比有验证集划分的 Agent v3 多用了 \~900 个训练节点
3. **FP 公式选择**: 平滑式 + 每步重归一化 优于 APPNP 式
   - 平滑式: `f_new = (1-a)*f_old + a*(Adj @ f_old)` + 重归一化
   - APPNP式: `f_new = (1-a)*(Adj @ result) + a*X_orig` → 表现更差(min 跌到 2.2%)
4. **软集成策略**: FP集成(70%) + 已有Agent集成(30%) = 软概率加权平均
   - 比硬投票更稳定, 保留了概率信息

## 数据分析

- 7650节点, 512维特征, 8类, 6120训练/1530测试
- 测试节点平均度19.50 vs 训练26.36（度分布差异显著）
- 有向图 → 转无向 (A+A^T)
- 标签分布：0(5.0%), 1(22.2%), 2(9.3%), 3(12.0%), 4(11.4%), 5(10.9%), 6(25.1%), 7(4.2%)

## 核心技术栈

### 特征传播 (Feature Propagation)

```python
# 平滑式, K=5, alpha=0.5, 每步重归一化
f_prop = f_dense.copy()
for k in range(K):
    f_prop = (1 - alpha) * f_prop + alpha * (norm_adj @ f_prop)
    # 重归一化
    m = f_prop.mean(axis=0); s = f_prop.std(axis=0)
    s[s==0] = 1.0; f_prop = (f_prop - m) / s
```

### 类别权重 (核心突破)

```python
# 完全逆频率权重, 不是 sqrt!
cls_counts = np.array([(labels[train_idx]==c).sum() for c in range(NUM_CLASSES)])
cls_weights = 1.0 / (cls_counts / cls_counts.sum())
cls_weights = cls_weights / cls_weights.mean()  # 均值=1
# 结果: {0:1.8, 1:0.4, 2:0.96, 3:0.75, 4:0.79, 5:0.82, 6:0.36, 7:2.13}
```

### 全数据训练 + SWA

```python
# 无验证集, 无早停, 纯 SWA
for ep in range(1, epochs+1):
    # ... 正常训练 ...
    if ep >= 250:  # SWA 开始
        swa_n += 1
        # 累积平均
model.load_state_dict(swa_s)  # 最终使用 SWA 权重
```

### 软集成

```python
# 概率层面加权平均, 而非硬投票
combined = w_ours * our_probs[train_idx] + (1-w_ours) * existing_probs
# w_ours=0.70 最优 (FP 主导, 已有集成补充)
```

## 配置参数

- GCN: 2层, hidden\_dim=256, dropout=0.5, DropEdge=0.5
- lr=0.02 (AdamW), weight\_decay=5e-4
- CosineAnnealing LR schedule
- Label smoothing: 0.1
- SWA: epoch 250-500
- epoch=500

## 最终结果

### 软集成 (70% FP + 30% Agent v3)

- 预测分布:
  \| 类别 | 预测% | 训练% | 差异 |
  \|------|-------|-------|------|
  \| C0 | 4.8% | 5.0% | -0.2% |
  \| C1 | 22.2% | 22.2% | 0.0% |
  \| C2 | 8.5% | 9.3% | -0.8% |
  \| C3 | 12.7% | 12.0% | +0.7% |
  \| C4 | 11.6% | 11.4% | +0.2% |
  \| C5 | 9.6% | 10.9% | -1.3% |
  \| C6 | 25.1% | 25.1% | 0.0% |
  \| C7 | 5.6% | 4.2% | +1.4% |
- min=4.8%, max=25.1%
- 偏置校准几乎为零: {C0:0.0, C1:-0.1, C6:-0.1, C7:0.0}

### FP 单独 (20种子, 不加Agent集成)

- min=6.1%, max=20.3% — 极其平衡, 甚至略过平衡

## 失败尝试

1. **kNN 图增强**: 加了 67k 条特征相似边(权重0.3), 反而降低分数
2. **APPNP 式 FP**: 表现比平滑式更差
3. **sqrt 逆频率权重**: 太弱, min 只有 3.5%
4. **更多投票版本(7版本硬投票)**: min=2.9%, 远不如软集成

## 未来改进方向

1. **更多 FP 种子**: 20→50-100 可进一步提升稳定性
2. **不同 K 值集成**: K=3, K=5, K=7 分别训练后软集成
3. **更深网络**: 3层 GCN + 残差连接
4. **warmup + cosine**: 学习率调度改进
5. **结合度感知验证集**: 也许全数据+度感知早停可以兼顾两者优势

## 代码位置

- workspace/code/cls/fp\_classweight\_ensemble.py — 完整训练脚本
- workspace/code/cls/eval\_utils.py — 度感知划分+评估工具
- workspace/outputs/classification/fp\_classweight\_probs.npz — 20种子软概率(可复用于集成)

## 输出文件

- workspace/outputs/classification/B1.csv
- workspace/outputs/classification/fp\_classweight\_probs.npz (20种子概率)

