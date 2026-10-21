# 分类任务完整实验记录 — 2025-03-20

## 数据分析发现

- B1.npz: 7650节点, 512维特征, 8类, 6120训练/1530测试
- 图是有向的，无自环，无孤立节点
- 特征已标准化（均值0，标准差1），100%稠密
- 标签严重不平衡：类别0(5.0%), 类别7(4.2%), 类别6(25.1%)
- 无向化后边数382308，平均度50
- 测试节点平均度19.5 < 训练节点26.36

## 实验流程

### 阶段一：架构筛选（seed=42）
| 排名 | 架构 | Val Acc | 备注 |
|------|------|---------|------|
| 1 | GCN | 0.5531 | DropEdge+SWA+LS |
| 2 | SGC | 0.5057 | K=2 |
| 3 | GIN | 0.4902 | SWA+LS |
| 4 | GraphSAGE | 0.4314 | |
| 5 | APPNP | 0.3423 | |
| 6 | GAT | 0.2492 | 崩溃！lr太高导致全坍缩为类别5 |

### 阶段二：网格搜索调参
- **GCN最佳**: lr=0.01, hidden_dim=256, dropout=0.5, drop_edge=0.3, wd=5e-4, ls=0.1 → **val=0.5621**
- SGC最佳: lr=0.05, K=3, wd=1e-4 → val=0.5229
- GIN最佳: lr=0.005, hidden_dim=64, dropout=0.7, wd=1e-4 → val=0.5474
- GAT修正: lr=0.002, drop_edge=0.0, ls=0.15 → val=0.5098（降低lr是关键！GAT对lr极其敏感）

### 阶段三：多种子集成（GCN, 10种子 seeds 42-51）
- 种子val: mean=0.5555, std=0.0042, min=0.5474, max=0.5637
- 集成预测分布严重偏向多数类（类别7仅1.1%, 类别0仅2.9%）

### 阶段四：后处理
- 额外加5种子(seeds 52-56)，共15种子，mean=0.5564
- 标签传播尝试失败（预测坍缩为类别6）
- 偏置校准成功：
  - 偏置: 类别0=+0.8, 类别1=-0.6, 类别6=-0.8, 类别7=+0.286
  - 最终分布: {0:168, 1:259, 2:120, 3:234, 4:234, 5:176, 6:278, 7:61}
  - 最少类4.0%, 最多类18.2%，分布健康
- 图结构特征增强无帮助（val=0.5613 vs 0.5621）

## 有效发现
1. GCN在此数据集上表现最佳，超越GAT/GIN等复杂架构
2. GAT需要极低lr（~0.002），高lr直接崩溃
3. DropEdge对GCN稳定有效（0.3最佳）
4. 偏置校准是将少数类从0.7%提升到4%的关键
5. 标签传播在此数据集上不可靠

## 失败记录
- GAT阶段一崩溃（lr=0.005太高）
- 标签传播（K=5, alpha=0.9）导致预测坍缩
- 图结构特征（度数/PageRank/K-core）无帮助
- SGC的K>2时分数下降，但lr=0.05对K=3有效

## 关键洞察
- B榜图小(7650节点)，简单的GCN+DropEdge效果最好
- 偏置校准是必须的——模型天然偏向多数类
- 校准需约束少数类不超过训练占比的3倍，防止过度校准

## 代码位置
- workspace/code/cls/analyze_data.py — 数据分析
- workspace/code/cls/stage1_arch_search.py — 阶段一架构筛选
- workspace/code/cls/stage2_grid_search.py — 阶段二网格搜索
- workspace/code/cls/stage3_multi_seed.py — 阶段三种子集成
- workspace/code/cls/stage4b_bias_calibrate.py — 阶段四偏置校准
- workspace/code/cls/stage4c_struct_features.py — 结构特征实验
- workspace/code/cls/stage4d_extra_seeds.py — 额外种子+最终校准

## 输出文件
- workspace/outputs/classification/B1.csv — 最终提交
