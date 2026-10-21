"""
Pi-Agent CLI — Layer 3 入口
============================
与 main.py 相同的命令行接口，启动 PiAgent 执行任务。

用法:
    python -m pi_agent.main --task classification --data workspace/data/cls/B1.npz
    python -m pi_agent.main --task recommendation --data workspace/data/rec/
    python -m pi_agent.main --task all
    python -m pi_agent.main --update-online --online 0.409 --task classification
"""
import argparse
import os
import sys
import json


def _check_output_sanity(task_type: str, output_dir: str, bench: str = "A") -> bool:
    """Quick sanity check on output CSV."""
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
            labels = df["label"].values
            if len(set(labels)) < 2:
                return False
        else:
            if "prediction" not in df.columns:
                return False
            empty_ratio = df["prediction"].isna().mean() + (df["prediction"] == "").mean()
            if empty_ratio > 0.5:
                return False
        return True
    except Exception:
        return False


def run_task(task_type: str, data_path: str, output_dir: str,
             bench: str = "A",
             budget: int = None, fresh_start: bool = False):
    """启动单个任务。"""
    from pi_agent.agent import PiAgent

    print(f"\n{'='*60}")
    print(f"  Task: {task_type}  |  Architecture: Pi-Agent")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    agent = PiAgent(
        task_type=task_type,
        output_dir=output_dir,
        bench=bench,
        budget=budget,
        fresh_start=fresh_start,
    )

    crashed = False
    try:
        agent.run()
        print(f"\n[OK] Pi-Agent: {task_type} completed successfully.")
    except KeyboardInterrupt:
        print(f"\n[ABORT] User interrupted {task_type}", file=sys.stderr)
        crashed = True
    except Exception as e:
        print(f"\n[FAIL] Pi-Agent: {task_type} crashed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        crashed = True

    # Safety net
    try:
        if not _check_output_sanity(task_type, output_dir, bench):
            csv_name = f"{bench}1.csv" if task_type == "classification" else f"{bench}2.csv"
            print(f"\n⚠️  Output {csv_name} may be invalid!")
    except Exception as e:
        print(f"  [SafetyNet] Validation failed: {e}", file=sys.stderr)

    if crashed:
        sys.exit(1)


def update_online(task: str, online_score: float, output_dir: str):
    """Update best score tracker with confirmed online score."""
    from glob import glob

    tag = "cls" if task == "classification" else "rec"
    score_path = os.path.join(output_dir, f"_best_{'st' if task == 'classification' else 'rec'}_score.json")
    key_local = "st_val" if task == "classification" else "best_ndcg"

    run_dirs = sorted(glob(os.path.join(output_dir, f"*_{tag}")), reverse=True)
    if not run_dirs:
        print(f"[Update] No run folder found.")
        return
    latest_dir = run_dirs[0]

    prev_info = {}
    if os.path.exists(score_path):
        with open(score_path) as f:
            prev_info = json.load(f)

    prev_online = prev_info.get("online", 0)
    print(f"[Update] {task}: online {prev_online:.4f} → {online_score:.4f}")

    if online_score > prev_online:
        prev_info.update({"online": online_score, key_local: prev_info.get(key_local, 0), "dir": latest_dir})
        with open(score_path, "w") as f:
            json.dump(prev_info, f)
        print(f"  ✅ _best updated: online={online_score:.4f}")
    else:
        print(f"  📦 online not improved, _best unchanged (online={prev_online:.4f})")


def main():
    parser = argparse.ArgumentParser(description="AFAC2026 AutoML Agent — Pi-Agent Architecture")
    parser.add_argument("--task", choices=["classification", "recommendation", "all"], default="all")
    parser.add_argument("--update-online", action="store_true")
    parser.add_argument("--online", type=float, default=None)
    parser.add_argument("--cls-data", default="workspace/data/cls/A1.npz")
    parser.add_argument("--rec-data", default="workspace/data/rec/")
    parser.add_argument("--output", default="workspace/outputs/")
    parser.add_argument("--bench", choices=["A", "B"], default="A")
    parser.add_argument("--fresh", action="store_true", help="Force fresh start, ignore checkpoint")
    parser.add_argument("--budget", type=int, default=None, help="Override time budget in seconds")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.update_online:
        if args.online is None:
            print("Error: --online is required with --update-online", file=sys.stderr)
            sys.exit(1)
        update_online(args.task, args.online, args.output)
        sys.exit(0)

    if args.task == "all":
        run_task("classification", args.cls_data, args.output, args.bench, args.budget, args.fresh)
        run_task("recommendation", args.rec_data, args.output, args.bench, args.budget, args.fresh)
    elif args.task == "classification":
        run_task("classification", args.cls_data, args.output, args.bench, args.budget, args.fresh)
    elif args.task == "recommendation":
        run_task("recommendation", args.rec_data, args.output, args.bench, args.budget, args.fresh)
    else:
        parser.print_help()

    print("\n=== Pi-Agent: All tasks completed ===")


if __name__ == "__main__":
    main()
