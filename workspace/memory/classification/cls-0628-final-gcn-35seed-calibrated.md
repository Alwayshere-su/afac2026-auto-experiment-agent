# GCN 35种子 + 激进偏置校准 — B榜分类最终实验

## 日期
2024-06-28

## 核心结果

**最终 B1.csv**: workspace/outputs/classification/B1.csv

| 指标 | 值 |
|------|------|
| 架构 | GCN（手工MyGCNConv，scatter_add消息传递） |
| 种子数 | 35（seeds 42-76） |
| 集成方式 | softmax概率平均 |
| 后处理 | 全类别激进偏置校准 |
| 验证集acc（校准后） | 0.4812 |
| 验证集acc（校准前） | 0.4812 |

**预测分布（全部 ≥ 5%）：**

| 类别 | 预测数 | 占比 |
|------|--------|------|
| 0 | 107 | 7.0% |
| 1 | 422 | 27.6% |
| 2 | 126 | 8.2% |
| 3 | 142 | 9.3% |
| 4 | 144 | 9.4% |
| 5 | 141 | 9.2% |
| 6 | 366 | 23.9% |
| 7 | 82 | 5.4% |

## 配置

| 参数 | 值 |
|------|------|
| hidden_dim | 512 |
| lr | 0.003 |
| dropout | 0.3 |
| num_layers | 2 |
| dropedge_rate | 0.2 |
| weight_decay | 1e-4 |
| epochs | 300 |
| patience | 50 |
| warmup_epochs | 30 |
| cosine_tmax | 400 |
| SWA | 开启（后半段） |
| LayerNorm | 开启 |
| Label Smoothing | 0.0 |
| 优化器 | AdamW |
| 调度器 | CosineAnnealingLR |

## 偏置校准参数

| 类别 | bias |
|------|------|
| 0 | +0.90 |
| 1 | 0.00 |
| 2 | +0.10 |
| 3 | 0.00 |
| 4 | 0.00 |
| 5 | +0.05 |
| 6 | -0.30 |
| 7 | +0.75 |

## 实验过程

### 阶段一：架构筛选（已完成，参照前序实验）
- GCN 0.4534 > GAT 0.4273 > GraphSAGE 0.4126 > SGC 0.3775 > APPNP 0.3415 > GIN 0.2508

### 阶段二：单种子调参（已完成，参照前序实验）
- GCN 最优: hd=512, lr=0.003, val_acc=0.4779

### 本轮新增实验

#### GAT 重新验证
- 10种子 GAT（hd=256, lr=0.003）：均值 0.4528，类别7完全坍塌（0预测）
- 结论：GAT在此有向图上确实不如GCN，不是配置问题

#### APPNP 网格搜索
- 搜索60个配置（K×alpha×hd）
- 最佳：K=2, alpha=0.05, hd=128, val_acc=0.4542
- 10种子APPNP：均值 0.4498，类别7几乎坍塌（2预测）
- 结论：APPNP不如GCN

#### GCN+GAT+APPNP 跨架构集成
- GCN+GAT (20模型)：分布变差，类别7仅0.4%
- GCN+GAT+APPNP (30模型)：分布更差
- 结论：GAT/APPNP污染集成，纯GCN最优

#### Focal Loss 测试
- gamma=0.25: 0.4739, gamma=0.5: 0.4729, gamma=1.0: 0.4680, gamma=2.0: 0.4629
- 基准NLL: 0.4763
- 结论：Focal Loss持续降分，无效

#### 标签传播后处理
- 最佳LP配置: alpha=0.3, iter=1, val_acc=0.4984
- 但预测分布严重偏向多数类：类别7仅5个(0.3%)
- LP+Bias: val_acc=0.5057，但类别7仅23(1.5%)
- 结论：LP对分布有害

#### hd=768 测试
- 15种子 hd=768 均值: 0.4783（vs hd=512 均值 0.4757）
- 有轻微提升但差异在1个std内
- 混合集成后分布改善有限

### 阶段三+四：35种子集成 + 激进偏置校准

最终采用35种子 hd=512 GCN + 全类别激进偏置校准。

校准策略（三轮）：
1. Round 1: 少数类(b0, b7)粗搜索，要求 val class0≥55 & class7≥45
2. Round 2: 所有8类精细搜索
3. Round 3: 激进加bias，接受val_acc小幅下降换更好分布

最终偏置使类别0从50→107，类别7从21→82，所有类别≥5.4%。

## 有效发现

1. **GCN是此有向图唯一有效架构**：GAT和APPNP都无法正确预测少数类（类别7完全或几乎坍塌）
2. **偏置校准是关键后处理**：通过验证集搜索每类bias，可显著改善少数类预测
3. **Focal Loss在此数据集上无效**：所有gamma值都降分
4. **标签传播偏向多数类**：使分布更极端
5. **hd=768有轻微提升**但不足以改变集成结果
6. **跨架构集成有害**：弱模型（GAT/APPNP）污染强模型（GCN）

## 失败记录

- GAT 10种子集成：均值0.4528，类别7=0
- APPNP 10种子集成：均值0.4498，类别7=2
- 跨架构集成（GCN+GAT/APPNP）：分布恶化
- Focal Loss：持续降分
- 标签传播：分布极端化
- 图结构特征增强（前序实验）：降分

## 关键洞察

1. 此有向图数据上GCN是唯一可靠架构，GAT和APPNP的注意力/传播机制在有向图上不稳定
2. 少数类预测是系统性挑战，偏置校准是目前唯一有效的缓解手段
3. 更多种子（35 vs 20）提升了集成稳定性

## 代码位置

- workspace/code/cls/gat_verify_and_appnp.py — GAT验证+APPNP网格搜索+跨架构集成
- workspace/code/cls/test_focal_loss.py — Focal Loss测试
- workspace/code/cls/gcn_final_ensemble.py — GCN 15种子+偏置校准
- workspace/code/cls/gcn_25seeds_calibrate.py — GCN 25种子+偏置校准+缓存
- workspace/code/cls/label_propagation_post.py — 标签传播后处理
- workspace/code/cls/aggressive_calibrate.py — 激进偏置校准（最终使用）
- workspace/code/cls/gcn_hd768_extra.py — hd=768测试+追加种子
- workspace/code/cls/gcn_hd768_ensemble.py — hd=768 15种子集成

## 输出文件

- **workspace/outputs/classification/B1.csv** — 最终提交文件
- workspace/outputs/classification/test_probs_cache.pkl — 35种子缓存
- workspace/outputs/classification/B1_base_25seed.csv — 25种子基础版
- workspace/outputs/classification/B1_lp.csv — 标签传播版
- workspace/outputs/classification/B1_lp_bias.csv — LP+Bias版
