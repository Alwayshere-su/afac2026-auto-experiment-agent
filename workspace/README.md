# workspace 工作区

Agent 和固化预测器的运行资产。所有命令都需要在**项目根目录**执行，因为代码里使用的是 `workspace/...` 相对路径。

## 目录

| 路径 | 内容 | 是否随仓库分发 |
|---|---|---|
| `code/cls/` | 分类预测器 `predict.py`、验证工具、`final_ensemble.npz`（53 种子集成概率）、`fp_20seeds.npz`（FP 20 种子缓存） | 是 |
| `code/rec/` | 推荐预测器 `predict.py`、NDCG/HitRate 验证工具 | 是 |
| `outputs/recommendation/seed*/` | 11 个训练目录，每个含 `best_model.pt`；其中 10 个被 `predict.py` 的 `MODEL_DIRS` 使用，`seed42_20260722_171809` 只用于提供 `holdout_uids.npy` | 是（约 130 MB） |
| `outputs/classification/` | 分类预测结果目录，**当前不存在**，运行 `code/cls/predict.py` 时自动创建 | — |
| `logs/` | `trajectory_B1.json`（分类 103 轮）、`trajectory_B2.json`（推荐 165 轮）实验轨迹 | 是 |
| `memory/` | Agent 跨轮次实验记忆（分类 13 篇 + 索引、推荐 2 篇 + 索引） | 是 |
| `feedback/` | 线上反馈：分类 Accuracy 0.40915、推荐 NDCG@10 0.23683 | 是 |
| `references/` | Agent 的方法知识库（分类/推荐/数据与训练规范） | 是 |
| `data/` | **官方数据，当前未放入** | 否 |

## 数据放置

按根目录 `README.md` 的约定：

```text
workspace/data/cls/B1.npz
workspace/data/rec/train.csv
workspace/data/rec/test.csv
workspace/data/rec/user.csv
workspace/data/rec/item.csv
```

官方数据体积大且不可公开分发，所以不在仓库里。没有这些文件时，预测器无法端到端运行；代码语法、模型权重和路径完整性仍可检查。

## 两个预测器

```powershell
# 分类：FP + 类权重 GCN 20 种子 + 53 种子软集成
python workspace/code/cls/predict.py --data workspace/data/cls/B1.npz --output B1.csv

# 推荐：10 模型 GRU4Rec+UserFeat+BlendNet 集成 + 分桶后处理
python workspace/code/rec/predict.py --data workspace/data/rec --output B2.csv
```

推荐预测器会读取 `workspace/outputs/recommendation/seed42_20260722_171809/holdout_uids.npy`，因此这个目录不能删；`MODEL_DIRS` 里的另外 10 个目录也不能重命名。
