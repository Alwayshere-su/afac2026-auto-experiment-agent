# Agent 实验记忆 — recommendation

- [B榜首次实验](rec-0722-B榜首次实验.md) — 7模型集成+分桶后处理，本地holdout NDCG=0.1459。架构dedup+CE+feat+blendnet, lr=1.5e-3, do=0.4。cw触顶(3.0-5.0)，后处理增益0.0072。贝叶斯收缩/冷启动pop blend均无效。小模型优于大模型、高LR+高DO协同有效。
- [B榜第二轮](rec-0722-B榜第二轮.md) — 10模型集成+分桶后处理，本地holdout NDCG=0.1484（无PP=0.1423）。do=0.5>0.4，lr=1.5e-3+wd=5e-5最优。cw仍触顶(5.0)，冷启动增强无效。单模型holdout从~0.125→0.1336。

