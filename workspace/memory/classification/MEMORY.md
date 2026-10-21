# Agent Experiment Memory — classification

- [FP+类权重集成](cls-20250723-fp-classweight.md) — **线上最优 0.409**。核心技术：FP(K=5, alpha=0.5)特征传播 + 全逆频率类权重 + 20种子全数据训练(无验证集划分) + SWA，与Agent 53种子（final\_ensemble.npz）软集成（w=0.7:0.3）。类权重让min class从2.5%→6.1%，是LB突破0.40的关键突破。lr=0.02优于之前的0.01。
- [改进版v3：特征传播+大规模集成](cls-20250320-improved-v3.md) — 62 GCN种子集成，K=5传播特征(512维)+6维图结构特征。关键发现：lr=0.02新最佳学习率(mean=0.4639 vs lr=0.01的0.4606)，特征传播比原始特征提升\~0.005 val，对低度测试节点尤其重要。偏置{0:+0.4, 1:-0.4, 6:-0.6, 7:+0.4}。GIN可达0.4641，GAT仍然不行(0.41-0.43)。
- [多源FP+度加权集成](cls-20250723-multisource-degw.md) — 线上 0.403。多K值(K=3,5,7)特征传播 + 多架构(GCN/GAT/SAGE)集成，度加权平均。GAT在传播特征下达0.43，SAGE达0.42。
- [Improved v2: 180种子](cls-20250723-improved-v2-180seeds.md) — 大规模Multi-K + Multi-Arch集成，探索集成规模上限。发现种子数超过\~60后边际收益趋零。
- [v4/v5：多样性K值+类别平衡](cls-20250320-v4-v5.md) — 不同K值(3,5,7)FP + 类别权重 + 温度缩放实验。K=3/5/7各有优势，跨K值集成略优。
- [完整四阶段实验(v2)](cls-20250320-stage1-4-v2.md) — 度感知划分重跑。GCN val=0.4657但线上仅0.4000，gap=0.066。发现测试节点平均度19.50 vs 训练26.36，低度节点表现差是gap主因。GCN >> GAT >> GraphSAGE/SGC。
- [完整四阶段实验(原始)](cls-0320-stage1-4-full.md) — 首次完整四阶段运行，确定GCN为最优架构。GCN(0.4502) >> GAT(0.3995) > SGC(0.3742)。有向图必须转无向图。
- [C7抑制器](cls-20250724-c7suppressor.md) — C7类别专项优化，尝试不同偏置和权重策略。
- [B榜完整四阶段](cls-0628-b榜-full-pipeline.md) — B榜首轮，验证A榜方法可迁移性。GCN仍然最优，B榜数据量更小(7650节点/512特征 vs A榜13752/767)。
- [GCN最终实验](cls-0628-gcn-final.md) — GCN全流程总结，包含所有阶段最优配置。
- [GCN完整pipeline](cls-0628-gcn-full-pipeline.md) — 阶段一至三详细记录，DropEdge=0.5最优，SWA稳定提升。
- [GCN Multi-Seed Ensemble](cls-0628-gcn-ensemble.md) — 15种子集成+偏置校准。保守偏置不够，需要同时boost少数类(C0,C7)和penalty多数类(C1,C6)。
- [GCN 35种子+激进偏置](cls-0628-final-gcn-35seed-calibrated.md) — 35种子最终实验，偏置范围扩大到\[-1.5, +1.5]，分布匹配训练集。

