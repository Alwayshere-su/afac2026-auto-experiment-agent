# GCN 阶段一至三完整实验记录

## 数据分析发现
- B榜: 7650节点, 512维特征, 8类, 6120训练/1530测试
- **图完全有向**（双向边=0），**无自环**，最小度数=8（无孤立节点）
- **特征已标准化**（均值0，标准差1），100%稠密
- **标签分布极度不均**：类别0(5.0%), 1(22.2%), 2(9.3%), 3(12.0%), 4(11.4%), 5(10.9%), 6(25.1%), 7(4.2%)
- 按合规铁律：有向图原样使用，不加自环/不无向化

## 关键问题与解决
- **PyG GCNConv 内部 gcn_norm 对边做去重导致 edge_weight 不匹配**：手工实现 GCNConv（scatter_add 消息传递），手动计算对称归一化边权重
- **GAT 搜索时 hidden=256+heads=8 疑似 OOM**：精简搜索空间

## 阶段一：架构筛选（单种子 seed=42）

| 架构 | val_acc | 配置要点 |
|------|---------|---------|
| **GCN** | **0.4534** | lr=0.01, hidden=128, dropout=0.5, de=0.2, ls=0.05 |
| GAT | 0.4273 | lr=0.005, hidden=128, heads=4, de=0.2, ls=0.1 |
| GraphSAGE | 0.4126 | lr=0.01, hidden=128, dropout=0.5 |
| SGC | 0.3775 | lr=0.01, K=2 |
| APPNP | 0.3415 | lr=0.01, K=10, alpha=0.1 |
| GIN | 0.2508 | lr=0.01, hidden=128 (严重欠拟合) |

前3名: GCN, GAT, GraphSAGE

## 阶段二：网格搜索调参

### GCN 最优
- **val_acc = 0.4779**
- lr=0.005, hidden_dim=256, dropout=0.3, dropedge_rate=0.2, label_smoothing=0.0, weight_decay=1e-4, num_layers=2, LayerNorm=True
- 搜索了 36+27=63 个配置
- 关键发现：label_smoothing=0.0 反而更好（对此有向图数据集）；lr=0.005 优于 0.01

### GAT 最优
- **val_acc = 0.4755**
- lr=0.003, hidden_dim=256, dropout=0.3, heads=4, dropedge_rate=0.2, label_smoothing=0.05, weight_decay=5e-4
- 搜索了 45 个配置
- 关键发现：GAT 对 dropout=0.7 极敏感（val≈0.27）；lr=0.003 最优

### GraphSAGE 最优
- **val_acc = 0.4632**
- lr=0.01, hidden_dim=64 (!), dropout=0.7, label_smoothing=0.05, weight_decay=1e-4
- 搜索了 39 个配置
- 关键发现：GraphSAGE 在此有向图上不如 GCN/GAT；小 hidden_dim+高 dropout 意外有效

### 阶段二结论：GCN 0.4779 > GAT 0.4755 > SAGE 0.4632，选 GCN 进阶段三

## 阶段三：GCN 多种子集成

- 8 个种子 (seed 42-49)，GCN 最优配置
- val_acc: [0.4820, 0.4739, 0.4730, 0.4698, 0.4641, 0.4632, 0.4673, 0.4592]
- 均值 0.4691 ± 0.0068
- 筛选 5/8 种子（>0.4657），均衡集成（softmax 概率平均）
- 预测分布: [55, 423, 127, 141, 145, 136, 482, 21]

## 有效发现
1. **手工 GCNConv 绕过 PyG 的边处理 bug**是关键——PyG 对有向图的 gcn_norm 内部去重会导致 edge_weight 不匹配
2. **label_smoothing=0.0 对有向图 GCN 最优**——与参考文件建议可能不同，需根据实际数据调整
3. **特征已标准化**，无需额外预处理
4. GAT 与 GCN 在此数据集上表现接近（差距仅 0.0024）

## 输出文件
- workspace/outputs/classification/B1.csv

## 代码位置
- workspace/code/cls/stage1_arch_search.py — 阶段一架构筛选
- workspace/code/cls/stage2_gcn_grid.py — GCN 网格搜索
- workspace/code/cls/stage2_gat_grid.py — GAT 网格搜索（初版，OOM）
- workspace/code/cls/stage2_gat_v2.py — GAT 精简搜索
- workspace/code/cls/stage2_sage_grid.py — GraphSAGE 网格搜索
- workspace/code/cls/stage3_gcn_ensemble.py — 多种子集成+生成B1.csv
- workspace/code/cls/analyze_data.py — 数据分析
