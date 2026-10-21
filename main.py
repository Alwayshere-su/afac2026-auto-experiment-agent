"""
AFAC2026 自动化实验 Agent — 主入口
====================================
接收命令行参数，启动 Pi-Agent 系统，在 2 小时预算内自主完成多轮实验。
支持分类任务（图节点分类）和推荐任务（序列推荐），可单独或依次运行。

用法示例:
    python main.py --task classification --cls-data B1.npz --output outputs/ --bench B
    python main.py --task recommendation --rec-data data/ --output outputs/ --bench B
    python main.py --task all --cls-data B1.npz --rec-data data/ --output outputs/ --bench B
"""
import argparse
import os
import sys


def _check_output_sanity(task_type: str, output_dir: str, bench: str = "A") -> bool:
    """输出文件完整性快速检查。

    不依赖真实标签，仅验证输出文件格式是否合理（非空、类别数 > 1 等）。
    返回 True 表示输出文件存在且格式正确。
    """
    import pandas as pd

    csv_name = f"{bench}1.csv" if task_type == "classification" else f"{bench}2.csv"
    csv_path = os.path.join(output_dir, csv_name)

    if not os.path.exists(csv_path):
        return False

    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            return False

        if task_type == "classification":
            # 分类任务：至少要有 2 个不同类别（全为同一类说明模型崩溃）
            labels = df["label"].values
            if len(set(labels)) < 2:
                return False
        else:
            # 推荐任务：prediction 列不能大面积为空
            if "prediction" not in df.columns:
                return False
            empty_ratio = df["prediction"].isna().mean() + (df["prediction"] == "").mean()
            if empty_ratio > 0.5:
                return False

        return True
    except Exception:
        return False


def _ensure_best_output(task_type: str, output_dir: str, bench: str = "A"):
    """输出安全网：验证输出质量，异常时尝试从最佳存档恢复。

    Agent 从零开始构建模型，不使用预训练权重。如果输出文件损坏或缺失，
    从历史最佳运行目录复制 CSV 作为最后的兜底方案。
    """
    import json
    from glob import glob

    tag = "cls" if task_type == "classification" else "rec"
    key = "st_val" if task_type == "classification" else "best_ndcg"
    score_path = os.path.join(output_dir, f"_best_{'st' if task_type == 'classification' else 'rec'}_score.json")

    if not os.path.exists(score_path):
        return

    with open(score_path) as f:
        best_info = json.load(f)
    best_val = best_info.get(key, 0)
    best_dir_name = os.path.basename(best_info.get("dir", "unknown"))

    # 第一层防护：输出文件缺失或异常
    if not _check_output_sanity(task_type, output_dir, bench):
        csv_name = f"{bench}1.csv" if task_type == "classification" else f"{bench}2.csv"
        reason = "missing" if not os.path.exists(os.path.join(output_dir, csv_name)) else "degenerate"
        print(f"\n⚠️  Output {csv_name} is {reason}!")
        print(f"   🛡️  Best archive: {best_dir_name} ({key}={best_val:.4f})")
        return

    # 第二层防护：检查轨迹日志中是否有模型崩溃标记
    run_dirs = sorted(glob(os.path.join(output_dir, f"*_{tag}")), reverse=True)
    if not run_dirs:
        return

    latest_dir = run_dirs[0]
    traj_files = glob(os.path.join(latest_dir, "trajectory_*.json"))
    if not traj_files:
        return

    try:
        with open(traj_files[0]) as f:
            traj = json.load(f)
    except Exception:
        return

    if isinstance(traj, dict):
        entries = traj.get("entries", [])
    elif isinstance(traj, list):
        entries = traj
    else:
        return

    finalize_entries = [e for e in entries if e.get("phase") == "finalize"]
    if not finalize_entries:
        return

    if finalize_entries[-1].get("config", {}).get("collapsed", False):
        print(f"\n⚠️  Latest run COLLAPSED (best archive: {best_dir_name}, {key}={best_val:.4f})")

    # 最后兜底：从最佳存档复制 CSV
    if not _check_output_sanity(task_type, output_dir, bench):
        csv_name = f"{bench}1.csv" if task_type == "classification" else f"{bench}2.csv"
        import shutil
        best_dir = best_info.get("dir", "")
        src = os.path.join(best_dir, csv_name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, csv_name))
            print(f"  🛡️ Last resort: copied {csv_name} from {best_dir_name}")


def run_task(task_type: str, data_path: str, output_dir: str,
             bench: str = "A", budget: int = None, fresh_start: bool = False):
    """运行单个任务（分类或推荐）。

    创建 PiAgent 实例并启动自主实验循环。Agent 在预算内完成架构搜索、
    超参调优、多种子集成和后处理，最终输出预测 CSV 和过程日志。

    参数:
        task_type: "classification" 或 "recommendation"
        data_path: 数据文件/目录路径
        output_dir: 输出根目录
        bench: A 榜或 B 榜
        budget: 时间预算（秒），默认从配置读取 7200
        fresh_start: 是否强制忽略已有 checkpoint
    """
    from pi_agent.agent import PiAgent

    print(f"\n{'='*60}")
    print(f"  Task: {task_type}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    brain = PiAgent(task_type, output_dir=output_dir,
                    bench=bench, budget=budget, fresh_start=fresh_start)

    crashed = False
    try:
        brain.run()
        print(f"\n[OK] Task {task_type} completed successfully.")
    except KeyboardInterrupt:
        print(f"\n[ABORT] User interrupted {task_type}", file=sys.stderr)
        crashed = True
    except Exception as e:
        print(f"\n[FAIL] Task {task_type} crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        crashed = True

    # 安全网：无论是否崩溃，都验证输出质量
    try:
        _ensure_best_output(task_type, output_dir, bench)
    except Exception as e:
        print(f"  [SafetyNet] Validation failed: {e}", file=sys.stderr)

    if crashed:
        sys.exit(1)


def update_online(task: str, online_score: float, output_dir: str):
    """更新线上分数追踪器和知识库。

    参数:
        task: "classification" 或 "recommendation"
        online_score: 线上评测分数
        output_dir: 输出根目录
    """
    import json
    from glob import glob

    tag = "cls" if task == "classification" else "rec"
    score_path = os.path.join(output_dir, f"_best_{'st' if task == 'classification' else 'rec'}_score.json")
    key_local = "st_val" if task == "classification" else "best_ndcg"

    run_dirs = sorted(glob(os.path.join(output_dir, f"*_{tag}")), reverse=True)
    if not run_dirs:
        print(f"[Update] No run folder (*_{tag}) found. Run the task first.")
        return
    latest_dir = run_dirs[0]

    prev_info = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            prev_info = json.load(f)
    local_val = prev_info.get(key_local, 0)

    prev_online = prev_info.get("online", 0)
    print(f"[Update] {task}: online {prev_online:.4f} → {online_score:.4f}")

    if online_score > prev_online:
        # 线上分数提升：更新最佳记录
        prev_info.update({"online": online_score, key_local: local_val, "dir": latest_dir})
        with open(score_path, "w") as f:
            json.dump(prev_info, f)
        print(f"  ✅ _best更新: online={online_score:.4f}")

        # 同步更新知识库
        kb_path = os.path.join(output_dir, f"_research_knowledge_{tag}.json")
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
            kb["best_result"]["online"] = online_score
            if "run_history" in kb and kb["run_history"]:
                latest = kb["run_history"][-1]
                if "提交后确认" in latest.get("note", ""):
                    latest["note"] = f"✅ 已确认 online={online_score:.4f}"
            kb["last_updated"] = __import__('time').strftime("%Y-%m-%d %H:%M:%S")
            with open(kb_path, "w", encoding="utf-8") as f:
                json.dump(kb, f, ensure_ascii=False, indent=2)
    elif online_score <= prev_online:
        # 线上分数未提升：标记回退
        print(f"  📦 online未提升, _best保持不变 (online={prev_online:.4f})")
        kb_path = os.path.join(output_dir, f"_research_knowledge_{tag}.json")
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = json.load(f)
            if "run_history" in kb and kb["run_history"]:
                latest = kb["run_history"][-1]
                if "提交后确认" in latest.get("note", ""):
                    latest["note"] = f"❌ 已回退 online={online_score:.4f} ≤ best={prev_online:.4f}"
            kb["last_updated"] = __import__('time').strftime("%Y-%m-%d %H:%M:%S")
            with open(kb_path, "w", encoding="utf-8") as f:
                json.dump(kb, f, ensure_ascii=False, indent=2)


def main():
    """命令行入口。解析参数后启动一个或两个任务。"""
    parser = argparse.ArgumentParser(description="AFAC2026 AutoML Agent")
    parser.add_argument("--task", choices=["classification", "recommendation", "all"], default="all",
                       help="任务类型：classification(分类) / recommendation(推荐) / all(全部)")
    parser.add_argument("--update-online", action="store_true",
                       help="更新线上分数（需配合 --online 使用）")
    parser.add_argument("--online", type=float, default=None,
                       help="线上评测分数")
    parser.add_argument("--cls-data", default="workspace/data/cls/A1.npz",
                       help="分类任务数据文件路径 (.npz)")
    parser.add_argument("--rec-data", default="workspace/data/rec/",
                       help="推荐任务数据目录路径")
    parser.add_argument("--output", default="workspace/outputs/",
                       help="输出根目录")
    parser.add_argument("--bench", choices=["A", "B"], default="A",
                       help="A 榜或 B 榜")
    parser.add_argument("--fresh", action="store_true",
                       help="强制从头开始，忽略已有 checkpoint")
    parser.add_argument("--budget", type=int, default=None,
                       help="时间预算（秒），默认从配置文件读取 7200")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 仅更新线上分数模式
    if args.update_online:
        if args.online is None:
            print("Error: --online is required with --update-online", file=sys.stderr)
            sys.exit(1)
        update_online(args.task, args.online, args.output)
        sys.exit(0)

    # 依次运行任务（禁止并行，比赛规则要求单进程）
    if args.task == "all":
        run_task("classification", args.cls_data, args.output, args.bench, args.budget, args.fresh)
        run_task("recommendation", args.rec_data, args.output, args.bench, args.budget, args.fresh)
    elif args.task == "classification":
        run_task("classification", args.cls_data, args.output, args.bench, args.budget, args.fresh)
    elif args.task == "recommendation":
        run_task("recommendation", args.rec_data, args.output, args.bench, args.budget, args.fresh)
    else:
        parser.print_help()

    print("\n=== All tasks completed ===")


if __name__ == "__main__":
    main()
