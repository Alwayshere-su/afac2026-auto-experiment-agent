# GCN 最终完整实验 — B榜分类任务

## 数据分析
- B榜: **7650节点, 512维特征, 8类**, 6120训练/1530测试
- **图完全有向**（双向边=0），**无自环**，最小度数=8（无孤立节点）
- **特征100%稠密，已标准化**（均值0，标准差1）
- **标签分布不均**：0(5.0%), 1(22.2%), 2(9.3%), 3(12.0%), 4(11.4%), 5(10.9%), 6(25.1%), 7(4.2%)
- 按合规铁律：有向图原样使用，不加自环/不无向化

## 最终输出
- **B1.csv**：workspace/outputs/classification/B1.csv
- 预测分布：[50, 426, 115, 151, 148, 140, 480, 20]，类别7最少(1.3%)，类别6最多(31.4%)

## 最终最优配置

| 参数 | 值 |
|------|------|
| 架构 | GCN（手工MyGCNConv，scatter_add消息传递） |
| num_layers | 2 |
| hidden_dim | **512** |
| lr | **0.003** |
| dropout | 0.3 |
| dropedge_rate | 0.2 |
| weight_decay | 1e-4 |
| epochs | 300 |
| patience | 50 |
| warmup_epochs | 30 |
| cosine_tmax | 400 |
| SWA | 开启（后半段） |
| LayerNorm | 开启 |
| Label Smoothing | 0.0（不用） |
| 优化器 | AdamW |
| 调度器 | CosineAnnealingLR |

## 阶段一：架构筛选（单种子 seed=42）

| 架构 | val_acc | 备注 |
|------|---------|------|
| **GCN** | **0.4534** | 手工Conv，有向图对称归一化 |
| GAT | 0.4273 | PyG GATConv, add_self_loops=False |
| GraphSAGE | 0.4126 | PyG SAGEConv |
| SGC | 0.3775 | K=2 |
| APPNP | 0.3415 | K=10, alpha=0.1（可能配置不当） |
| GIN | 0.2508 | 严重欠拟合 |

前3名: GCN, GAT, GraphSAGE

## 阶段二：网格搜索调参

### GCN 最优
- val_acc = 0.4779（单种子seed=42）
- lr=0.005, hidden_dim=256, dropout=0.3, dropedge_rate=0.2, ls=0.0, wd=1e-4, layers=2, LayerNorm=True
- 搜索63个配置

### GAT 最优
- val_acc = 0.4755（单种子seed=42）
- lr=0.003, hidden_dim=256, dropout=0.3, heads=4, dropedge_rate=0.2, ls=0.05, wd=5e-4
- 但14种子集成后均值仅0.4510，说明单种子结果不可靠

### GraphSAGE 最优
- val_acc = 0.4632（单种子seed=42）
- lr=0.01, hidden_dim=64, dropout=0.7, ls=0.05
- 14种子集成均值0.4557

## 阶段三：多种子集成演进

| 轮次 | 配置 | 种子数 | 均值 | 备注 |
|------|------|--------|------|------|
| v1 | hd=256, lr=0.005 | 8 | 0.4691 | 原始GCN |
| v2 | hd=256, lr=0.005 | 14 | 0.4692 | 增加种子 |
| v3 | hd=256, lr=0.005, 类别加权 | 3 | 0.4357 | **严重降分，放弃** |
| v4 | hd=256, lr=0.005 | 20 | 0.4663 | 种子过多反而降分 |
| **v5** | **hd=512, lr=0.003** | **16** | **0.4750** | **补充调参发现新最优** |
| **v6** | **hd=512, lr=0.003** | **20** | **0.4753** | **最终版** |

关键发现：**hd=512 + lr=0.003 是最优组合**，3种子均值0.4779，20种子均值0.4753。

## 阶段四：后处理尝试

| 方法 | 结果 |
|------|------|
| 标签传播（训练节点one-hot） | val_acc虚高0.9992（标签泄露），预测分布更差 |
| 标签传播（修正版，全图模型概率） | alpha=0.5, iter=2, val_acc=0.4730，但预测分布更差(类别7仅0.3%) |
| 图结构特征增强（度数/PageRank等6维） | val_acc降低（0.4749→0.4665） |
| 温度缩放 | 不改变argmax，无效 |
| 跨配置融合（3配置×14种子=42模型） | 分布类似，类别7从18→20 |

## 有效发现

1. **手工GCNConv绕过PyG bug是关键**：PyG对有向图的gcn_norm内部去重导致edge_weight不匹配
2. **hd=512 + lr=0.003 是最优组合**：补充调参中发现的，比原最优(hd=256,lr=0.005)高0.005
3. **DropEdge=0.2至关重要**：de=0.0时val_acc降至0.4439（-0.03）
4. **3层GCN不如2层**：3层+hd=128持平(0.4725)，但3层+hd=256/512均下降
5. **GAT在此有向图上表现差**：即使修复epochs=400配置，14种子均值仅0.4510，类别7完全坍塌
6. **类别加权loss大幅降低准确率**：0.4730→0.4357
7. **Label Smoothing=0.0对有向图GCN最优**：与参考文件建议可能不同
8. **种子质量差异大**：最佳0.4861 vs 最差0.4665，差距0.02

## 失败记录

- GAT 14种子集成：均值0.4510，类别7完全坍塌到0%
- GraphSAGE 14种子集成：均值0.4557
- 图结构特征：降低准确率
- 类别加权loss：严重降低准确率
- 标签传播（修正版）：预测分布更极端

## 关键洞察

1. 此有向图数据集上**GCN是最佳架构**，GAT表现差（可能因为多头注意力在稀疏有向图上不稳定）
2. **hidden_dim=512是关键突破**：更大的隐层维度显著提升泛化能力
3. 少数类（类别0和7）预测不足是系统性问题，多种方法尝试无效
4. **手工GCNConv实现正确**，PyG的GCNConv在有向图上有bug

## 代码位置
- workspace/code/cls/stage1_arch_search.py — 阶段一6架构对比
- workspace/code/cls/stage2_gcn_grid.py — GCN网格搜索
- workspace/code/cls/stage2_gat_v2.py — GAT网格搜索
- workspace/code/cls/stage2_sage_grid.py — GraphSAGE网格搜索
- workspace/code/cls/stage3_gcn_14seeds_lp.py — GCN 14种子+LP（旧）
- workspace/code/cls/stage3_gcn_20seeds_balanced.py — 类别加权测试
- workspace/code/cls/stage3_gcn_final.py — GCN 20种子+LP
- workspace/code/cls/stage3_gcn_hd512.py — GCN hd=512 16种子
- workspace/code/cls/stage3_gcn_20final.py — **GCN hd=512 最终20种子**
- workspace/code/cls/stage3_gat_ensemble.py — GAT 14种子（epochs=300，失败）
- workspace/code/cls/stage3_gat_fixed.py — GAT 14种子（epochs=400）
- workspace/code/cls/stage3_sage_ensemble.py — GraphSAGE 14种子
- workspace/code/cls/stage3_fusion.py — 跨配置融合
- workspace/code/cls/gcn_extra_tuning.py — GCN补充调参（发现hd=512最优）
- workspace/code/cls/test_graph_features.py — 图结构特征测试
- workspace/code/cls/quick_test_cw.py — 类别加权快速测试
- workspace/code/cls/analyze_data.py — 数据分析

## 输出文件
- **workspace/outputs/classification/B1.csv** — 最终提交文件（20种子GCN hd=512）
- workspace/outputs/classification/B1_base_ensemble.csv — 14种子hd=256基础版
- workspace/outputs/classification/B1_hd512.csv — 16种子hd=512版
- workspace/outputs/classification/B1_fusion.csv — 跨配置融合版
- workspace/outputs/classification/B1_gat_ensemble.csv — GAT集成版
- workspace/outputs/classification/B1_sage.csv — GraphSAGE集成版
- workspace/outputs/classification/B1_lp.csv — 标签传播版
