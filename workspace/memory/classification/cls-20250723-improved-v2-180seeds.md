# 分类任务实验记忆 — Improved v2: Multi-K + Multi-Arch 180种子集成

## 日期
2025-07-23

## 实验目标
在FP+类权重基础上（在线0.40719），通过多K值特征传播、多架构集成、更多种子来缩小与排行榜最高分(0.43)的差距。

## 核心技术栈

### 1. 多K值特征传播集成
- K=3, alpha=0.5: 25 GCN seeds
- K=5, alpha=0.5: 55 GCN seeds (25 from v2 + 10 extra + 20 extra)
- K=5, alpha=0.3: 10 GCN seeds
- K=5, alpha=0.7: 10 GCN seeds
- K=7, alpha=0.5: 25 GCN seeds
- 总计额外25 GCN seeds (alpha多样性)

### 2. 多架构互补
- GCN: 145 seeds (主要)
- GAT: 15 seeds (hidden=128, heads=4, lr=0.01, de_ratio=0.3)
- GraphSAGE: 15 seeds (hidden=256, lr=0.02, de_ratio=0.3)
- 总计: 180 seeds

### 3. 训练配置
- 完全逆频率类别权重: {0:1.8, 1:0.4, 2:0.96, 3:0.75, 4:0.79, 5:0.82, 6:0.36, 7:2.13}
- 全数据训练(无验证集) + SWA (epoch 250-500)
- Warmup 30 epochs + Cosine LR
- Label smoothing: 0.1
- DropEdge: 0.5 (GCN), 0.3 (GAT/GraphSAGE)
- AdamW, weight_decay=5e-4
- 500 epochs

### 4. 特征工程
- Feature Propagation (平滑式, 每步重归一化)
- 6维图结构特征: log(1+in_deg), log(1+out_deg), log(1+deg), PageRank, log(1+2hop), feature_norm
- 特征标准化(原始+图结构分别标准化后拼接)

### 5. 偏置校准
- 在测试集分布上搜索偏置(匹配训练分布)
- 最终偏置: {0:-0.15, 1:-0.05, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.1, 7:-0.2}
- 优化目标: min(class%) >= 4%, minimize L1距离到训练分布

## 最终结果

### 测试集预测分布
| 类别 | 预测% | 训练% | 差异 |
|------|-------|-------|------|
| C0 | 4.9% | 5.0% | -0.1% |
| C1 | 20.1% | 22.2% | -2.1% |
| C2 | 9.7% | 9.3% | +0.4% |
| C3 | 13.3% | 12.0% | +1.4% |
| C4 | 11.5% | 11.4% | +0.1% |
| C5 | 10.6% | 10.9% | -0.4% |
| C6 | 25.3% | 25.1% | +0.2% |
| C7 | 4.6% | 4.2% | +0.4% |
- min=4.6%, max=25.3%

### 各阶段演进
| 种子数 | C1% | C6% | C7% | 说明 |
|--------|-----|-----|-----|------|
| 90 (v2) | 16.4% | 20.5% | 8.6% | GCN(K=3,5,7)+GAT |
| 125 | 20.7% | 22.1% | 5.5% | +GCN(alpha=0.3,0.7)+GraphSAGE |
| 155 | 22.0% | 22.7% | 5.6% | +30 GCN(K=5) |
| 175 | 22.1% | 23.2% | 4.1% | +20 Deep GCN(3层) |
| 180 | 20.1% | 25.3% | 4.6% | +25 GCN(K=5) — 最终 |

### 标签传播尝试
- LP alpha=0.1-0.3: 少数类崩溃(min<1.4%)
- LP+raw blend: C1/C6过度膨胀
- 结论: LP不适用于此数据集，保持纯集成

### 与previous ensemble融合
- final_ensemble.npz (53 seeds, C0=2.7%, C1=34%): 即使5-10%权重也会恶化分布
- 结论: 不融合旧集成，纯新180种子最优

## 关键发现

1. **多K值FP增加多样性**: K=3,5,7分别捕获不同尺度的图结构信息，集成后更稳定
2. **多架构互补有效**: GAT和GraphSAGE提供与GCN互补的信号，改善少数类预测
3. **alpha多样性**: alpha=0.3/0.5/0.7各自产生不同的平滑效果
4. **180种子足够**: 从125→180的边际改善递减，但仍有小幅分布对齐改善
5. **标签传播无效**: 在此数据集上LP使分布崩溃，不应使用
6. **类权重是关键**: 完全逆频率权重使得偏置校准需求极低

## 失败尝试
1. 标签传播 — 少数类崩溃
2. 与final_ensemble.npz融合 — 分布恶化
3. Deep GCN (3层+残差) — 边际改善极小，不划算

## 最终提交版本 (195 seeds, SGC inclusive)
- **架构**: GCN 145 + GAT 15 + GraphSAGE 15 + SGC 15 + Deep GCN (部分)
- **分布**: C0:5.0%, C1:20.3%, C2:9.9%, C3:13.1%, C4:11.6%, C5:10.4%, C6:25.1%, C7:4.6%
- **偏置**: {0:-0.15, 1:-0.05, 2:0.0, 3:0.0, 4:0.0, 5:0.0, 6:0.1, 7:-0.2}
- 195 seeds in final submission; 210-seed version also competitive

## 代码位置
- workspace/code/cls/improved_ensemble_v2.py — 主训练脚本 (90 seeds: GCN K=3,5,7 + GAT)
- workspace/code/cls/supplementary_seeds.py — 补充训练 (35 seeds: GCN alpha=0.3,0.7 + GraphSAGE)
- workspace/code/cls/extra_30_seeds.py — 额外GCN (30 seeds)
- workspace/code/cls/deep_gcn_seeds.py — Deep GCN (20 seeds, not saved separately)
- workspace/code/cls/final_25_seeds.py — 额外25 GCN seeds
- workspace/code/cls/final_30_seeds.py — 额外30 GCN seeds (210 total)
- workspace/code/cls/large_hidden_seeds.py — 20 GCN hidden=512 seeds
- workspace/code/cls/sgc_seeds.py — 15 SGC seeds + final merge (195 seeds)
- workspace/code/cls/final_merge.py — 180-seed merge
- workspace/code/cls/final_blend.py — Blend/LP experiments (LP无效)
- workspace/code/cls/refine_bias_v2.py — Bias calibration experiments

## 输出文件
- workspace/outputs/classification/B1.csv (最终提交)
- workspace/outputs/classification/final_155_probs.npz
- workspace/outputs/classification/extra_25_probs.npz
- workspace/outputs/classification/supplementary_probs.npz
- workspace/outputs/classification/improved_v2_probs.npz
