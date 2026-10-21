# AFAC2026 自动化实验 Agent

## 项目概览

- 赛题与成绩：AFAC2026（阿里天池）自动化实验 Agent；A 榜综合 **0.63906**（54/360），B 榜综合 **0.32299**（第 27 名），决赛挑战组**三等奖（第三名）**。任务细节见下方“比赛背景”。
- 核心：Pi-Agent 在 2 小时预算内自主完成 103 轮（分类）/ 165 轮（推荐）实验；`workspace/code/` 保存赛后固化的两个预测器。
- 仓库内容：Agent 源码、预测器、10 个进入集成的推荐模型权重，另有 1 个早期 seed 目录提供 `holdout_uids.npy`（其权重未进入集成）；实验轨迹与记忆也一并保留。**官方数据不在仓库内**，按下方说明放入 `workspace/data/` 后才能端到端运行。
- 仓库只保留代码、预测器、模型权重与实验记录；比赛文档、路演 PPT 与内置 Git Bash 保留在本地，不随代码仓库分发。

## 比赛背景

AFAC2026（阿里天池）金融智能创新大赛的赛题是**自动化实验 Agent**：参赛系统需要在有限预算、稀疏反馈和禁止并行的约束下，自主完成“提出实验配置 → 运行训练 → 读取反馈 → 调整下一步策略”的多轮闭环，而不是提交一个预先写死的模型。

挑战组奖项：一等奖 1 名、二等奖 2 名、三等奖 3 名。

### 赛程与成绩

| 阶段 | 规则 | 分类 | 推荐 | 综合 | 排名/结果 |
|---|---|---|---|---|---|
| 初赛（A 榜） | 360 支队伍打榜 | 0.76845 | 0.50967 | **0.63906** | 第 54 名 |
| 复赛（B 榜） | 3 天，每天 3 次提交 | 0.40915 | 0.23683 | **0.32299** | 第 27 名 |
| 决赛 | 前 30 名提交复现材料 → 15 组入选答辩 → 最终 9 组到场 | — | — | — | 挑战组**三等奖（第三名）** |

> 综合分数为分类、推荐两项分数的算术平均。

### 两个子任务

| 子任务 | 任务形式 | 评测指标 |
|---|---|---|
| 产品分类 | 图节点分类 | Accuracy |
| 产品推荐 | 用户-产品二部图上的序列推荐 | NDCG@10 |

- 预算与约束：单任务 2 小时；实验串行执行；官方数据不随仓库分发。
- 本项目结果：Pi-Agent 在预算内完成 103 轮（分类）/ 165 轮（推荐）实验；固化预测器在 `workspace/code/`，过程轨迹在 `workspace/logs/`，实验记忆在 `workspace/memory/`。
- 官方数据需按“安装”一节放入 `workspace/data/` 后才能端到端运行；预测器本身可用仓库内权重直接复现 B 榜结果。
- 比赛文档、路演材料与 PPT 保留在本地，不随仓库分发；本 README 只保留运行所需的比赛背景摘要。

## 环境要求

- Python 3.10+
- NVIDIA GPU
- CUDA 12.x
- Windows 下建议安装 Git for Windows，Agent 的 shell 工具会优先使用系统 Git Bash；未安装时回退 `cmd.exe`。

> **重要**：所有命令需在项目根目录下运行。项目内全部使用相对路径，无需修改任何配置即可运行。

## 安装

```bash
pip install -r requirements.txt
```

> **测评时请将数据文件放置如下：**
> - 分类数据 `B1.npz` → `workspace/data/cls/B1.npz`
> - 推荐数据（`train.csv`, `test.csv`, `user.csv`, `item.csv`）→ `workspace/data/rec/`

## API Key 配置

在项目根目录创建 `.api_key` 文件，写入 DashScope API Key 即可：

```bash
echo "your_api_key" > .api_key
```

***

## 一、Agent 系统（完整自动化实验）

Agent 从零开始自主完成实验在 2 小时预算内不断迭代优化，最终输出预测结果。

### 分类任务

输入 `B1.npz` 图数据，Agent 自主训练 GCN/GAT/GraphSAGE 等多种架构，筛选最优模型，输出 1530 个测试节点的类别预测。

```bash
python main.py --task classification --cls-data workspace/data/cls/B1.npz --output workspace/outputs/ --bench B
```

**输出**：`workspace/outputs/classification/B1.csv` + `workspace/logs/trajectory_B1.json`

### 推荐任务

输入用户交互序列、用户特征、产品特征，Agent 自主训练 GRU4Rec+UserFeat+BlendNet 等模型，输出 10000 个测试用户的 Top-10 产品推荐列表。

```bash
python main.py --task recommendation --rec-data workspace/data/rec/ --output workspace/outputs/ --bench B
```

**输出**：`workspace/outputs/recommendation/B2.csv` + `workspace/logs/trajectory_B2.json`

### 两个任务依次运行

```bash
python main.py --task all --cls-data workspace/data/cls/B1.npz --rec-data workspace/data/rec/ --output workspace/outputs/ --bench B
```

***

## 二、预测器

预测器是 Agent 在实验过程中训练并筛选出的最优模型，已固化参数和权重，**无需 API Key，无需重新实验**，可直接运行复现 B 榜最优分数。

**适用场景**：快速复现分数、验证环境可用性、评审复现。

### 分类任务预测器

用 FP 特征传播 + 全逆频率类权重训练的 20 个 GCN 种子，与 Agent 的 53 种子软集成。首次运行需训练（约 15 分钟，结果自动缓存），后续运行秒级完成。

```bash
python workspace/code/cls/predict.py --data workspace/data/cls/B1.npz --output B1.csv
```

### 推荐任务预测器

加载 10 个预训练的 GRU4Rec+UserFeat+BlendNet 模型权重，等权集成后应用分桶后处理（共访矩阵增强 + 复购加权 + 流行度匹配）。纯推理，约 2 分钟。

```bash
python workspace/code/rec/predict.py --data workspace/data/rec/ --output B2.csv
```

***

## 三、输出文件格式

### B1.csv — 分类预测结果

```csv
test_idx,label
9,7
14,6
21,4
```

- `test_idx` 必须与官方 `sample_submission.csv` 一致
- `label` 为 0\~7 整数类别编号
- 共 1531 行（含表头）

### B2.csv — 推荐预测结果

```csv
uid,prediction
u000001,"i020992,i000001,i000003,i000004,i000005,i000006,i000007,i000008,i000009,i000010"
```

- `uid` 必须与官方 `test.csv` 一致
- `prediction` 为 10 个 iid，英文逗号分隔，按置信度降序排列
- 共 10001 行（含表头）

### trajectory\_B1.json / trajectory\_B2.json — 实验过程日志

```json
{
  "task": "classification",
  "total_rounds": 103,
  "trajectory": [
    {
      "round": 1,
      "agent_thinking": "Agent 对当前局势的分析",
      "config": { "architecture": ["GCN"], "lr": "0.02" },
      "feedback": { "val_acc": 0.4502 },
      "strategy": "Agent 下一步优化方向",
      "tools_called": [{"tool": "start_shell", "command": "python train.py"}],
      "budget_remaining": 7195
    }
  ]
}
```

每轮记录包含：实验配置、验证集反馈、优化策略、调用的工具、剩余时间预算。

***

## 四、项目结构

```
2026AFAC/
├── README.md                        # 本文件：概览、环境、命令、输出格式
├── main.py                          # Agent 系统入口
├── requirements.txt                 # 依赖清单
├── LICENSE                          # Apache-2.0（作者代码）
├── NOTICE                           # 第三方素材授权说明
├── .gitignore / .gitattributes
├── pi_agent/                        # Agent 核心代码
│   ├── __init__.py
│   ├── agent.py                     #   Agent 主循环（感知-决策-执行）
│   ├── config.py                    #   Pi-Agent 配置（复出口全局常量）
│   ├── context.py                   #   上下文窗口管理与压缩
│   ├── events.py                    #   事件总线（模块间通信）
│   ├── llm.py                       #   Qwen API 调用抽象
│   ├── main.py                      #   Pi-Agent 入口（CLI 参数解析）
│   ├── messages.py                  #   消息格式定义
│   ├── prompts.py                   #   系统提示词（任务定义、约束规则）
│   ├── session.py                   #   会话管理与 checkpoint
│   ├── state_machine.py             #   实验阶段状态机
│   ├── tools.py                     #   工具定义与分发
│   └── _tools_impl.py               #   工具底层实现
├── utils/                           # 全局配置与工具
│   ├── __init__.py
│   ├── config.py                    #   API Key 解析、预算、模型配置
│   ├── budget_tracker.py            #   时间预算追踪
│   └── data_loader.py               #   数据加载工具
└── workspace/                       # 工作区，详见 workspace/README.md
    ├── README.md
    ├── data/                        # 官方数据（当前未放入）
    │   ├── cls/                     #   B1.npz
    │   └── rec/                     #   train/test/user/item.csv
    ├── code/
    │   ├── cls/                     # 分类预测器
    │   │   ├── predict.py           #   主预测脚本
    │   │   ├── final_ensemble.npz   #   Agent 训练的 53 种子集成概率
    │   │   ├── fp_20seeds.npz       #   FP 20 种子预测缓存
    │   │   └── eval_utils.py        #   度感知划分、准确率计算
    │   └── rec/                     # 推荐预测器
    │       ├── predict.py           #   主预测脚本
    │       └── eval_utils.py        #   NDCG/HitRate 计算、序列截断
    ├── outputs/
    │   ├── classification/          # 分类输出（运行后生成，当前不存在）
    │   └── recommendation/          # 11 个 seed 目录（10 个进入集成 + 1 个提供 holdout）
    ├── references/                  # 方法参考文献
    │   ├── classification_methods.md
    │   ├── recommendation_methods.md
    │   └── data_and_training.md
    ├── memory/                      # Agent 跨轮次实验记忆
    │   ├── classification/          #   分类任务实验记录
    │   └── recommendation/          #   推荐任务实验记录
    ├── feedback/                    # 线上评测反馈记录
    └── logs/                        # 过程日志
        ├── trajectory_B1.json
        └── trajectory_B2.json
```

## 五、发布到 GitHub

仓库根目录已带 `.gitignore` 和 `.gitattributes`：

- `.gitignore` 排除 `.api_key`、`workspace/data/`、`docs/`、`vendor/bash/`、`__pycache__/` 和虚拟环境；官方数据与比赛材料不会入库。
- `.gitattributes` 把 `*.pt`、`*.npz`、`*.npy` 标记为二进制，避免换行转换。
- 仓库约 129 MB：模型权重约 128 MB（11 个 `best_model.pt`，每个约 12 MB），其余代码与实验记录约 1 MB。单文件都小于 GitHub 的 100 MB 限制，普通 Git 仓库即可；如果以后再加更大的权重，再考虑 Git LFS 或 Release 附件。
- 比赛文档、路演 PPT 与内置 Git Bash 只保留在本地，不随代码仓库分发。

