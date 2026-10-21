"""
Pi-Agent 核心 — 自主实验 Agent 主循环
======================================
基于 ReAct 模式（Think → Act → Observe）的自动化机器学习实验系统。

架构层次：
  Layer 1: LLMClient — Qwen API 调用抽象
  Layer 2: PiAgent  — 事件驱动 + 状态机 + 工具管线 + 会话持久化 + 上下文压缩

Agent 在 2 小时预算内自主完成四阶段流程：
  阶段1: 架构筛选（多种 GNN 架构水平对比）
  阶段2: 超参网格搜索（锁定最优架构，系统搜索 lr/hidden_dim/dropout 等）
  阶段3: 多种子集成（固定配置，变随机种子训练多个模型，等权平均）
  阶段4: 后处理（偏置校准 / 标签传播 / 共访矩阵增强等）

每轮循环：LLM 分析局势 → 决定行动 → 执行工具 → 观察结果 → 更新记忆 → 进入下一轮
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from pi_agent.events import (
    Event, EventBus,
    EVENT_AGENT_START, EVENT_AGENT_END,
    EVENT_TURN_START, EVENT_TURN_END,
    EVENT_TOOL_START, EVENT_TOOL_END,
    make_logging_listener,
)
from pi_agent.llm import LLMClient
from pi_agent.state_machine import StateMachine, AgentState
from pi_agent.session import SessionManager
from pi_agent.context import compress_messages
from pi_agent.tools import build_tool_manager


class PiAgent:
    """自主机器学习实验 Agent。

    核心设计：
      - 事件驱动：所有关键生命周期节点发出事件，便于日志记录和扩展
      - 状态机：RUNNING → TOOL_EXECUTING → RUNNING → ... → DONE，含状态钩子
      - 工具管线：define → register → intercept → execute → recycle
      - 会话管理：保存/恢复完整对话状态，支持中断后恢复
      - 上下文压缩：长对话自动压缩，保持在 LLM 上下文窗口内
      - 预算追踪：实时监控时间消耗，到期自动提醒收尾
    """

    def __init__(self, task_type: str,
                 output_dir: str = "workspace/outputs/", bench: str = "A",
                 budget: int = None, fresh_start: bool = False):
        """初始化 Agent。

        参数:
            task_type: "classification"（图节点分类）或 "recommendation"（序列推荐）
            output_dir: 输出根目录
            bench: "A" 或 "B" 榜
            budget: 时间预算（秒），默认从配置读取 7200（2小时）
            fresh_start: True 则删除已有 checkpoint，强制从头开始
        """
        self.task_type = task_type
        self.output_dir = output_dir
        self.bench = bench
        self._stop_requested = False

        # ── 时间预算追踪 ──
        from utils.budget_tracker import BudgetTracker
        from utils.config import TOTAL_BUDGET_SECONDS, SAFETY_MARGIN_SECONDS
        if budget is not None:
            self.budget = BudgetTracker(total_budget=budget, safety_margin=SAFETY_MARGIN_SECONDS)
        else:
            self.budget = BudgetTracker(total_budget=TOTAL_BUDGET_SECONDS, safety_margin=SAFETY_MARGIN_SECONDS)

        # ── 事件总线：解耦各模块通信 ──
        self.events = EventBus()
        self.events.on_any(make_logging_listener(self._print))

        # ── 状态机：管理 Agent 生命周期状态 ──
        self.state_machine = StateMachine()
        self._setup_state_hooks()

        # ── LLM 客户端：通过 Qwen API 进行推理决策 ──
        self.llm = LLMClient(print_fn=self._print)

        # ── 会话管理：支持中断恢复（checkpoint） ──
        self.session = SessionManager(task_type)

        # ── 实验记忆：记录每次实验的配置、结果、分析 ──
        self._memory_dir = Path(f"workspace/memory/{task_type}")
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        tag = "cls" if task_type == "classification" else "rec"
        self._memory_index = self._memory_dir / "MEMORY.md"
        self._protect_memory_index()

        # ── 工具管线：Agent 可调用的所有工具（读写文件、启停训练等） ──
        self._tool_manager, self._tool_handlers = build_tool_manager(
            task_type=task_type, bench=bench, memory_dir=self._memory_dir,
            print_fn=self._print, event_bus=self.events,
        )
        self._tool_handlers._on_stop = self._handle_stop
        self._tool_handlers._on_think = self._handle_think

        # ── 清理上次运行的临时文件 ──
        self._cleanup_workspace()

        if fresh_start:
            self.session.delete()

        # ── 核心运行时状态 ──
        self._user_goal = ""               # 用户目标描述
        self._compression_summary = ""     # 上下文压缩后的摘要
        self._last_experiment_done = False # 上一轮训练是否刚完成（触发反思）
        self._last_agent_thinking = ""     # Agent 最近一次的思考内容
        self._messages: List[Dict] = []    # LLM 对话历史
        self._recent_outputs: List[str] = [] # 最近的 shell 输出（用于异常检测）
        self._trajectory: List[Dict] = []  # 实验轨迹日志
        self._experiments_completed = 0    # 已完成实验计数
        tag2 = "cls" if task_type == "classification" else "rec"
        if self._memory_dir.exists():
            self._experiments_completed = len(list(self._memory_dir.glob(f"{tag2}-*.md")))

    # ═══════════════════════════════════════════════════════════
    # 初始化辅助方法
    # ═══════════════════════════════════════════════════════════

    def _setup_state_hooks(self):
        """在状态转移上挂载钩子函数。"""

        def _on_enter_running():
            pass

        def _on_enter_tool_executing():
            pass

        def _on_enter_done():
            self._print(f"\n🏁 Agent entering DONE state — finalizing...")

        self.state_machine.on_enter(AgentState.RUNNING, _on_enter_running)
        self.state_machine.on_enter(AgentState.TOOL_EXECUTING, _on_enter_tool_executing)
        self.state_machine.on_enter(AgentState.DONE, _on_enter_done)

    def _protect_memory_index(self):
        """MEMORY.md 损坏时从备份恢复。防止索引丢失导致历史实验无法读取。"""
        bak_path = self._memory_dir / "MEMORY.md.bak"
        if not self._memory_index.exists() or self._memory_index.stat().st_size < 50:
            if bak_path.exists() and bak_path.stat().st_size > 100:
                import shutil
                shutil.copy2(str(bak_path), str(self._memory_index))
                self._print(f"  🛡️ MEMORY.md restored from backup ({bak_path.stat().st_size} bytes)")
            elif not self._memory_index.exists():
                self._memory_index.write_text(f"# Agent Experiment Memory — {self.task_type}\n\n", encoding="utf-8")

    def _cleanup_workspace(self):
        """清理上次运行残留的临时文件和迭代脚本。"""
        import glob as _g
        import shutil
        for f in _g.glob("workspace/code/iteration_*.py"):
            try: os.remove(f)
            except OSError: pass
        tmp_dir = Path("workspace/outputs/.tmp")
        if tmp_dir.exists():
            try: shutil.rmtree(tmp_dir)
            except OSError: pass
            tmp_dir.mkdir(parents=True, exist_ok=True)

    def _handle_stop(self):
        """Agent 调用 stop 工具时触发：标记停止请求。"""
        self._stop_requested = True

    def _handle_think(self, topic: str) -> str:
        """Agent 调用 think 工具时触发：做一次纯文本深度推理。

        复用当前对话上下文，但不产生 tool call，让 LLM 对特定话题做深入分析。
        """
        think_msg = {
            "role": "user",
            "content": (
                f"[Deep Thinking]\n"
                f"Topic: {topic}\n\n"
                f"Analyze the current situation in depth. Consider:\n"
                f"- What do the existing results tell us?\n"
                f"- What hypotheses can we form? What's the underlying mechanism?\n"
                f"- What are the risks and alternatives?\n\n"
                f"Provide a thorough analysis, then conclude with a clear recommendation "
                f"for what to do next and WHY."
            )
        }
        messages = self._messages + [think_msg]
        result = self.llm.think(messages, max_tokens=1200)
        self._print(f"     💭 {result[:300]}...")
        return result

    # ═══════════════════════════════════════════════════════════
    # 系统提示词构建
    # ═══════════════════════════════════════════════════════════

    def _build_system_prompt(self) -> str:
        """构建榜别感知的系统提示词。

        从 prompts.py 读取模板，将 A 榜路径替换为当前榜别路径。
        """
        from pi_agent.prompts import CLS_SYSTEM_PROMPT, REC_SYSTEM_PROMPT
        tag = "cls" if self.task_type == "classification" else "rec"
        prompt = CLS_SYSTEM_PROMPT if tag == "cls" else REC_SYSTEM_PROMPT
        b = self.bench
        prompt = prompt.replace('A1.npz', f'{b}1.npz')
        prompt = prompt.replace('A1.csv', f'{b}1.csv')
        prompt = prompt.replace('A2.csv', f'{b}2.csv')
        return prompt

    # ═══════════════════════════════════════════════════════════
    # 主循环 — ReAct 模式 (Think → Act → Observe)
    # ═══════════════════════════════════════════════════════════

    def run(self) -> Dict:
        """启动自主实验循环。

        ReAct 模式流程：
          1. LLM 思考（Think）：分析当前状态 → 决定下一步行动
          2. 执行工具（Act）：运行训练脚本、读写文件、监控进程
          3. 观察结果（Observe）：读取训练输出 → 更新实验记忆
          4. 循环直到预算耗尽或 Agent 主动停止

        每轮记录写入 trajectory log，供评审核查实验过程。
        """
        self.budget.start()
        self.events.emit(Event(EVENT_AGENT_START, {"task": self.task_type, "bench": self.bench}))

        self._print(f"\n{'='*60}")
        self._print(f"🔬 Pi-Agent: {self.task_type}")
        self._print(f"{'='*60}")
        self._print(f"  Budget: {self.budget.total_budget}s | "
                    f"Model: {self.llm._active_provider.model} ({self.llm.active_provider_name})")
        self._print(f"  Architecture: Pi-Agent (event-driven + state machine + tool pipeline)")
        self._print(f"{'='*60}")

        # ── 加载 checkpoint 或全新开始 ──
        start_iter = 0
        if self.session.exists():
            ckpt = self.session.load()
            if ckpt:
                self._print(f"  🔄 Resuming from checkpoint: iter={ckpt['iteration']}")
                start_iter = ckpt["iteration"]
                self._compression_summary = ckpt.get("summary", "")
                self.budget.start_time -= ckpt.get("budget_elapsed", 0)
                self._messages = ckpt.get("messages", [])
                self._trajectory = ckpt.get("trajectory", [])
                self._experiments_completed = ckpt.get("experiments_completed", 0)
                self._print(f"      Restored {len(self._messages)} messages, {len(self._trajectory)} trajectory entries")

        # ── 构建初始消息 ──
        if start_iter == 0:
            # 首次启动：告诉 Agent 去哪里读记忆、反馈、参考文件
            self._user_goal = (
                f"Task: {self.task_type}.\n"
                f"📖 Step Zero (MUST do first on every run!):\n"
                f"1) read_file workspace/memory/{self.task_type}/MEMORY.md — check if experiments have been done before.\n"
                f"2) read_file workspace/feedback/{self.task_type}.md — check online scores and user suggestions\n"
                f"3) read_file workspace/references/{self.task_type}_methods.md — reference\n"
                f"4) list_files workspace/code/ and workspace/data/ — find existing scripts\n"
                f"⚠️ IF memory already has experiment records: start from the best known config.\n"
                f"⚠️ IF this is the first run (empty memory): follow the full four-phase pipeline.\n"
                f"All model code is written by you (write_file) and trained via start_shell.\n"
                f"Before finishing, write memory to workspace/memory/{self.task_type}/ and update MEMORY.md.\n"
            )
            start_iter = 1

        if not self._messages:
            system_prompt = self._build_system_prompt()
            # 提示 Agent 历史实验记忆的位置
            hints = []
            if self._memory_index.exists():
                hints.append(
                    f"Previous experiment memory is at `{self._memory_index}`."
                    f"**Please use the read_file tool to read it yourself**."
                )
            roadmap_path = Path(f"workspace/memory/{self.task_type}/exploration_roadmap.md")
            if roadmap_path.exists():
                hints.append(
                    f"Exploration roadmap is at `{roadmap_path}`."
                    f"**Please use read_file to read it**."
                )
            if hints:
                system_prompt += "\n\n## 🧠 Historical Experiment Memory\n" + "\n\n".join(hints) + "\n"

            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self._user_goal},
            ]

        # 上下文压缩后的摘要注入
        if self._compression_summary and start_iter > 1:
            self._messages.insert(1, {
                "role": "user",
                "content": f"[Previous round summary]\n{self._compression_summary}\n\nContinue experimenting; do not repeat work already done."
            })

        # ── 检查 API 可用性 ──
        if not self.llm.available:
            self._print(f"\n❌ No LLM provider available. Exiting.")
            self.events.emit(Event(EVENT_AGENT_END, {"reason": "no_api"}))
            return {}

        # ═══════════════════════════════════════════════════
        # ReAct 主循环
        # ═══════════════════════════════════════════════════
        self.state_machine.transition(AgentState.RUNNING)
        iteration = start_iter
        budget_final_warning_given = False

        while not self._stop_requested:
            # ── 预算耗尽检查 ──
            if self.budget.must_stop_now():
                if not budget_final_warning_given:
                    csv_name = f"{self.bench}1.csv" if self.task_type == "classification" else f"{self.bench}2.csv"
                    self._messages.append({
                        "role": "user",
                        "content": (
                            f"⏰ Budget exhausted, do NOT train new models. Please complete:\n"
                            f"1) Generate {csv_name} to workspace/outputs/{self.task_type}/\n"
                            f"2) write_file memory to workspace/memory/{self.task_type}/ and update MEMORY.md\n"
                            f"3) Call the stop tool to end the experiment\n"
                        )
                    })
                    budget_final_warning_given = True
                    self._print(f"\n⏰ Budget exhausted, waiting for Agent to wrap up...")

            iteration += 1
            elapsed_before = self.budget.elapsed()

            self.events.emit(Event(EVENT_TURN_START, {
                "iteration": iteration,
                "budget_elapsed": elapsed_before,
                "budget_remaining": self.budget.remaining(),
            }))
            self._print(f"\n─ Iteration #{iteration} | Elapsed {elapsed_before:.0f}s ─")

            # ── 反思触发：上一轮训练刚完成，强制 Agent 分析结果 ──
            if self._last_experiment_done:
                self._messages.append({
                    "role": "user",
                    "content": (
                        "[Reflection] The last training run just completed. Before starting anything new:\n"
                        "1. Did the result match your hypothesis? Why/why not?\n"
                        "2. What did you LEARN that you didn't know before?\n"
                        "3. Does your next planned experiment still make sense? Adjust strategy if needed.\n\n"
                        "Write a brief reflection to workspace/memory/classification/cls-reflection.md, then continue."
                    )
                })
                self._last_experiment_done = False

            # ── 阶段 1: LLM 调用（Think） ──
            t_api_start = time.time()
            content, reasoning, tool_calls_raw = self.llm.call_with_tools(self._messages)
            t_api_elapsed = time.time() - t_api_start

            remaining = self.budget.remaining()
            hint = " ⏳ Make full use of time" if remaining > 600 else " ⚠️ Running out of time, wrap up soon"
            self._print(f"     API time {t_api_elapsed:.1f}s | Remaining {remaining:.0f}s{hint}")

            if content is None and tool_calls_raw is None:
                self._print(f"  ❌ LLM call failed completely — stopping")
                self._stop_requested = True
                break

            # 显示 Agent 的思考内容
            thinking_text = content or reasoning or ""
            if thinking_text.strip():
                self._last_agent_thinking = thinking_text
            if content:
                self._print(f"\n{'─'*50}\n🧠 Agent:\n   {content}\n{'─'*50}")
            elif reasoning:
                self._print(f"\n{'─'*50}\n🧠 Agent:\n   {reasoning[:500]}\n{'─'*50}")

            # 将 LLM 响应加入对话历史
            assistant_msg = {
                "role": "assistant",
                "content": content or "",
                "reasoning_content": reasoning or "",
            }
            if tool_calls_raw:
                assistant_msg["tool_calls"] = tool_calls_raw
            self._messages.append(assistant_msg)

            if not tool_calls_raw:
                continue

            # ── 阶段 2: 执行工具（Act） ──
            self.state_machine.transition(AgentState.TOOL_EXECUTING)

            round_tools_list = []
            round_feedback = {}

            results = self._tool_manager.execute_sequential(tool_calls_raw)

            for tc_raw, result_str in results:
                fn = tc_raw.get("function", {})
                tool_name = fn.get("name", "?")

                self._fmt_tool_result(tool_name, result_str)

                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except Exception:
                    args = {}

                # 工具结果加入对话历史（LLM 下次调用时可见）
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": tc_raw.get("id", "call_0"),
                    "content": result_str,
                })

                # 记录工具调用摘要（用于轨迹日志）
                tool_summary = {"tool": tool_name}
                if tool_name == "read_file":
                    tool_summary["file"] = args.get("filepath", "")[:120]
                elif tool_name in ("write_file", "edit_file"):
                    tool_summary["file"] = args.get("file_path", args.get("filepath", ""))[:120]
                elif tool_name == "start_shell":
                    tool_summary["command"] = args.get("command", "")[:120]
                elif tool_name == "check_shell":
                    tool_summary["pid"] = args.get("pid", "?")
                elif tool_name == "kill_shell":
                    tool_summary["pid"] = args.get("pid", "?")
                elif tool_name == "stop":
                    tool_summary["action"] = "stop"
                round_tools_list.append(tool_summary)

                # 缓存 shell 输出用于异常检测
                if tool_name in ("check_shell", "run_shell"):
                    self._recent_outputs.append(str(result_str))
                    if len(self._recent_outputs) > 20:
                        self._recent_outputs = self._recent_outputs[-15:]

                # ── 检测实验是否完成 ──
                _is_experiment = False
                result_str_lower = str(result_str).lower()
                _has_metrics = any(kw in result_str for kw in (
                    "NDCG", "Accuracy", "Best", "Done", "最优", "全部完成"
                ))
                _has_error = any(kw in result_str_lower for kw in (
                    "traceback", "error:", "nameerror", "keyerror",
                    "attributeerror", "syntaxerror", "runtimeerror", "cuda error"
                ))
                if tool_name == "check_shell" and _has_metrics:
                    _is_experiment = True
                elif tool_name == "run_shell":
                    _is_experiment = _has_metrics and not _has_error
                if _is_experiment:
                    self._experiments_completed += 1
                    self._last_experiment_done = True  # 下一轮触发反思
                    self._print(f"  📊 Experiment #{self._experiments_completed} completed")

            # ── 构建轨迹日志条目 ──
            thinking = self._last_agent_thinking[:500] if self._last_agent_thinking else ""
            if not thinking and round_tools_list:
                tool_names = [t["tool"] for t in round_tools_list]
                thinking = f"[Tool calls: {', '.join(tool_names[:5])}]"

            # 从思考中提取策略摘要
            strategy = ""
            if thinking:
                sentences = [s.strip() for s in thinking.replace("\n", " ").split("。") if s.strip()]
                strategy = "。".join(sentences[-2:]) if len(sentences) >= 2 else (sentences[-1] if sentences else thinking[:200])

            round_config = self._extract_round_config(thinking_text, round_tools_list, round_feedback)

            self._trajectory.append({
                "round": len(self._trajectory) + 1,
                "iteration": iteration,
                "agent_thinking": thinking,
                "config": round_config if round_config else None,
                "feedback": round_feedback if round_feedback else None,
                "strategy": strategy if strategy else None,
                "tools_called": round_tools_list,
                "budget_remaining": int(self.budget.remaining()),
            })
            self._save_trajectory()

            self.events.emit(Event(EVENT_TURN_END, {
                "iteration": iteration,
                "tools_executed": len(round_tools_list),
            }))

            # ── 会话 checkpoint：支持中断后恢复 ──
            self.session.save(
                iteration=iteration, messages=self._messages,
                budget_elapsed=self.budget.elapsed(),
                summary=self._compression_summary,
                trajectory=self._trajectory,
                experiments_completed=self._experiments_completed,
            )

            self.state_machine.transition(AgentState.RUNNING)

            # ── 预算提醒：每 5 轮或达到阈值时通知 Agent ──
            remaining = self.budget.remaining()
            pct = remaining / self.budget.total_budget * 100
            if not hasattr(self, '_budget_pcts_seen'):
                self._budget_pcts_seen = set()
            budget_msg = None
            if iteration % 5 == 0:
                budget_msg = (
                    f"⏰ Budget status: {self.budget.elapsed():.0f}s used / {self.budget.total_budget}s total, "
                    f"{remaining:.0f}s remaining ({pct:.0f}%). Stop training when <300s remain."
                )
            for threshold in [50, 25, 10]:
                if pct < threshold and threshold not in self._budget_pcts_seen:
                    self._budget_pcts_seen.add(threshold)
                    budget_msg = (
                        f"⏰⚠️ Only {remaining:.0f}s ({pct:.0f}%) budget remaining!"
                        f"Must stop training, generate CSV, write memory, and stop when <300s remain."
                    )
                    break
            if budget_msg:
                self._messages.append({"role": "user", "content": budget_msg})
                self._print(f"  ⏰ {budget_msg}")

            # ── 上下文压缩：对话过长时自动压缩 ──
            _total_chars = sum(len(str(m.get("content", ""))) for m in self._messages)
            if _total_chars > 3_500_000:
                self._messages = compress_messages(
                    messages=self._messages,
                    trajectory=self._trajectory,
                    experiments_completed=self._experiments_completed,
                    compression_summary=self._compression_summary,
                    print_fn=self._print,
                )
                reminder = (
                    f"Context compressed. Please re-read_file {self._memory_index} to understand "
                    f"the global experiment state, and read specific memory files as needed. "
                    f"{self._experiments_completed} experiments completed."
                )
                self._messages.append({"role": "user", "content": reminder})

        # ── 收尾：保存轨迹、清理 session ──
        self._print(f"\n🏁 Experiment ended | {len(self._trajectory)} rounds")
        self.state_machine.transition(AgentState.DONE)
        self.events.emit(Event(EVENT_AGENT_END, {"total_rounds": len(self._trajectory)}))

        self._save_trajectory()
        self.session.delete()

        return self._find_output_csv()

    # ═══════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════

    def _fmt_tool_result(self, name: str, result: str):
        """格式化输出工具执行结果（避免刷屏）。"""
        lines = result.strip().split("\n")
        if name == "list_files":
            count = len([l for l in lines if l.strip()])
            self._print(f"     → Found {count} files")
            for line in lines[:8]:
                self._print(f"       {line}")
            if count > 8:
                self._print(f"       ... {count - 8} more files")
        elif name == "read_file":
            preview = result[:200].replace("\n", " ")
            self._print(f"     → {preview}...")
        elif name == "write_file" and "✅" in result:
            for line in result.strip().split("\n"):
                line = line.strip()
                if line:
                    self._print(f"     {line}")
        elif name == "start_shell":
            preview = result[:200].replace("\n", " ")
            self._print(f"     → {preview}")
        elif name == "check_shell":
            non_empty = [l for l in result.split("\n") if l.strip()]
            for line in non_empty[-6:]:
                self._print(f"       {line[:150]}")
        elif name == "kill_shell":
            preview = result[:200].replace("\n", " ")
            self._print(f"     → {preview}")
        elif name == "stop":
            self._print(f"     🛑 Experiment ended")
        else:
            preview = result[:200].replace("\n", " ")
            self._print(f"     → {preview}")

    def _save_trajectory(self):
        """持久化实验轨迹日志到磁盘。"""
        try:
            tag = f"{self.bench}1" if self.task_type == "classification" else f"{self.bench}2"
            path = f"workspace/logs/trajectory_{tag}.json"
            os.makedirs("workspace/logs", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "task": self.task_type,
                    "agent_type": "pi-agent",
                    "total_rounds": len(self._trajectory),
                    "trajectory": self._trajectory,
                }, f, indent=2, default=str, ensure_ascii=False)
        except Exception as e:
            self._print(f"  ⚠️ Trajectory save failed: {e}")

    def _find_output_csv(self) -> Dict:
        """查找 Agent 生成的输出 CSV 文件。"""
        csv_name = f"{self.bench}1.csv" if self.task_type == "classification" else f"{self.bench}2.csv"
        search_roots = [
            Path(self.output_dir),
            Path(f"workspace/outputs/{self.task_type}"),
        ]
        for root in search_roots:
            if root.exists():
                for csv_path in sorted(root.rglob(csv_name), reverse=True):
                    if csv_path.is_file():
                        self._print(f"  ✅ Found {csv_name} ({csv_path})")
                        return {}
        return {}

    @staticmethod
    def _extract_round_config(thinking_text: str, round_tools: list, round_feedback: dict) -> Optional[Dict]:
        """从 Agent 思考文本中提取结构化配置信息。

        通过正则匹配识别：架构名称、超参数值（lr/hidden_dim/dropout 等）、
        布尔开关（SWA/CosineAnnealing）、修改的文件名、训练反馈指标。
        这些信息用于轨迹日志中的 config 字段。
        """
        config = {}

        # ── 架构识别 ──
        _arch_map = [
            ("GAT", "GAT"), ("GCN", "GCN"), ("GraphSAGE", "GraphSAGE"),
            ("GIN", "GIN"), ("APPNP", "APPNP"), ("SGC", "SGC"),
            ("GRU4Rec", "GRU4Rec"), ("ResGCN", "ResGCN"),
        ]
        seen_archs = []
        for pat, aname in _arch_map:
            if re.search(r'\b' + pat + r'\b', thinking_text, re.IGNORECASE):
                seen_archs.append(aname)
        if seen_archs:
            config["architecture"] = seen_archs

        # ── 超参数提取（正则匹配） ──
        _hp_patterns = [
            (r'(?:hidden_dim|hd|hidden)[\s=:]*(\d+)', 'hidden_dim', int),
            (r'(?:num_layers|n_layers|layers)[\s=:]*(\d+)', 'num_layers', int),
            (r'(?:heads|num_heads|n_heads)[\s=:]*(\d+)', 'heads', int),
            (r'(?:dropout|do)[\s=:]*([\d.]+)', 'dropout', float),
            (r'(?:learning_rate|lr)[\s=:]*([\deE.+-]+)', 'lr', str),
            (r'(?:weight_decay|wd)[\s=:]*([\deE.+-]+)', 'weight_decay', str),
            (r'(?:label_smoothing|ls|label_smooth)[\s=:]*([\d.]+)', 'label_smoothing', float),
            (r'(?:epochs|num_epochs|n_epochs)[\s=:]*(\d+)', 'epochs', int),
        ]
        for pat, key, typ in _hp_patterns:
            m = re.search(pat, thinking_text, re.IGNORECASE)
            if m:
                try:
                    config[key] = typ(m.group(1))
                except (ValueError, TypeError):
                    pass

        # ── 布尔开关 ──
        for flag, key in [("SWA", "SWA"), ("CosineAnnealing", "cosine_annealing"),
                           ("grid_search", "grid_search")]:
            if re.search(flag, thinking_text, re.IGNORECASE):
                config[key] = True

        # ── 修改的文件名 ──
        touched = []
        for t in round_tools:
            if t.get("tool") in ("write_file", "edit_file"):
                fp = t.get("file", "")
                if fp and fp.endswith(".py"):
                    touched.append(fp)
        if touched:
            config["touched_files"] = touched

        if round_feedback:
            config["feedback"] = round_feedback

        return config if config else None

    def _print(self, msg: str):
        print(msg, flush=True)
