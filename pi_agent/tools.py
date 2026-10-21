"""
工具系统 — Pi-Agent Layer 2
============================
Agent 通过工具与外部环境交互。每个工具经过五步管线处理：

  1. DEFINE   — 定义工具 schema（名称、参数、描述）
  2. REGISTER — 注册工具名 → 处理函数的映射
  3. INTERCEPT — 执行前后钩子（可拦截或修改调用）
  4. EXECUTE   — 调用处理函数，捕获结果
  5. RECYCLE   — 执行后清理（如保存轨迹日志）

执行模式：
  - sequential（默认）：工具逐个执行，前一个结果对后续可见
  - parallel：工具并发执行（仅用于独立的读取操作）
"""
from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pi_agent.events import Event, EventBus, EVENT_TOOL_START, EVENT_TOOL_END

# ── Low-level tool implementations ──


class ToolManager:
    """
    Manages tool lifecycle: register → intercept → execute → recycle.

    Each tool handler receives (args: dict) and returns str.
    """

    def __init__(self, event_bus: EventBus = None, print_fn: Callable = None):
        self._handlers: Dict[str, Callable[[dict], str]] = {}
        self._before_hooks: List[Callable[[str, dict], Optional[str]]] = []
        self._after_hooks: List[Callable[[str, dict, str], str]] = []
        self._event_bus = event_bus
        self._print = print_fn or (lambda x: None)

    def register(self, name: str, handler: Callable[[dict], str]) -> None:
        """Register a tool handler function."""
        self._handlers[name] = handler

    def add_before_hook(self, hook: Callable[[str, dict], Optional[str]]) -> None:
        """
        Add a before-execution hook.
        Args:
            hook(tool_name, args) → None (allow) or str (block with this error message).
        """
        self._before_hooks.append(hook)

    def add_after_hook(self, hook: Callable[[str, dict, str], str]) -> None:
        """
        Add an after-execution hook.
        Args:
            hook(tool_name, args, result) → modified_result_str.
        """
        self._after_hooks.append(hook)

    # ── Execution ──

    def execute_sequential(self, tool_calls: List[Dict]) -> List[Tuple[Dict, str]]:
        """
        Execute tool calls one-by-one in order.
        Returns list of (tool_call_dict, result_string).
        """
        results: List[Tuple[Dict, str]] = []
        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "?")
            try:
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            except (json.JSONDecodeError, TypeError) as e:
                # Try JSON repair
                raw = fn["arguments"] if isinstance(fn["arguments"], str) else ""
                from pi_agent.llm import LLMClient
                repaired = LLMClient.repair_json(raw)
                if repaired != raw:
                    try:
                        args = json.loads(repaired)
                        fn["arguments"] = repaired
                        self._print(f"  🔧 JSON auto-repair succeeded")
                    except Exception:
                        args = {"_json_error": str(e)[:200], "_raw_preview": raw[:500], "_raw_len": len(raw)}
                else:
                    args = {"_json_error": str(e)[:200], "_raw_preview": raw[:500], "_raw_len": len(raw)}

            result = self._execute_one(name, args, tc)
            results.append((tc, result))
        return results

    def _execute_one(self, tool_name: str, args: dict, raw_tc: Dict) -> str:
        """Execute a single tool call with hooks."""
        t0 = time.time()

        # Emit start event
        if self._event_bus:
            self._event_bus.emit(Event(EVENT_TOOL_START, {
                "tool_name": tool_name,
                "tool_args_summary": self._fmt_args(tool_name, args),
                "tool_call_id": raw_tc.get("id", ""),
            }))

        # Before hooks
        for hook in self._before_hooks:
            block_msg = hook(tool_name, args)
            if block_msg is not None:
                return block_msg

        # Execute
        handler = self._handlers.get(tool_name)
        if handler:
            try:
                result = handler(args)
            except Exception:
                result = f"ERROR: {traceback.format_exc()}"
        else:
            result = f"Unknown tool: {tool_name}"

        # Truncate large outputs
        result_str = str(result)
        if len(result_str) > 250_000:
            result_str = result_str[:250_000] + "\n...[truncated]"

        # After hooks
        for hook in self._after_hooks:
            result_str = hook(tool_name, args, result_str)

        # Emit end event
        if self._event_bus:
            self._event_bus.emit(Event(EVENT_TOOL_END, {
                "tool_name": tool_name,
                "duration_ms": (time.time() - t0) * 1000,
                "result_len": len(result_str),
            }))

        return result_str

    @staticmethod
    def _fmt_args(name: str, args: dict) -> str:
        """Format tool args for display."""
        if name == "list_files":
            d = args.get("directory", "workspace")
            p = args.get("pattern", "**/*")
            return f"📂 {d}/{p}"
        elif name == "read_file":
            fp = args.get("filepath", "")
            fname = fp.split("/")[-1] if "/" in fp else fp
            return f"📖 {fname}"
        elif name == "write_file":
            fp = args.get("filepath", "")
            fname = fp.split("/")[-1] if "/" in fp else fp
            return f"✏️  {fname}"
        elif name == "run_shell":
            return f"💻 {args.get('command', '')[:80]}"
        elif name == "start_shell":
            return f"🚀 {args.get('command', '')[:80]}"
        elif name == "check_shell":
            return f"🔍 pid={args.get('pid', '?')}"
        elif name == "kill_shell":
            return f"💀 pid={args.get('pid', '?')}"
        elif name == "stop":
            return "🛑 stop"
        return f"🔧 {name}"


# ═══════════════════════════════════════════════════════════════
# Tool Handler Implementations
# ═══════════════════════════════════════════════════════════════

class ToolHandlers:
    """
    All tool handler implementations. Separated from ToolManager so the
    Agent can inject its own state (task_type, output_dir, memory_dir, etc.).
    """

    def __init__(self, task_type: str, bench: str = "A",
                 memory_dir: Path = None, print_fn: Callable = None):
        self.task_type = task_type
        self.bench = bench
        self.memory_dir = memory_dir or Path(f"workspace/memory/{task_type}")
        self._print = print_fn or (lambda x: None)
        # State hooks (set by Agent)
        self._on_stop: Optional[Callable[[], None]] = None
        self._on_think: Optional[Callable[[str], str]] = None  # (topic) → analysis

    # ── think ──

    def h_think(self, args: dict) -> str:
        """Deep reasoning tool — invokes the LLM without tools for analysis."""
        topic = args.get("topic", "")
        if not topic:
            return "❌ think tool requires a 'topic' parameter describing what to analyze."
        if not self._on_think:
            return "❌ Think backend not configured."
        self._print(f"  💭 Thinking about: {topic[:100]}...")
        result = self._on_think(topic)
        if result:
            return f"## Analysis: {topic}\n\n{result}"
        return "⚠️ Think completed but produced no output."

    # ── list_files ──

    def h_list_files(self, args: dict) -> str:
        from pi_agent._tools_impl import list_files
        directory = args.get("directory", "workspace")
        pattern = args.get("pattern", "**/*")
        result = list_files(directory, pattern)
        lines = [f"{r['path']} ({r['size_kb']}KB)" for r in result.get("files", [])]
        out = "\n".join(lines) if lines else "(empty)"
        total = result.get("total_found", len(result.get("files", [])))
        shown = result.get("count", len(result.get("files", [])))
        if result.get("overflow"):
            out += (f"\n\n⚠️ Truncated: found {total} files total, only showing first {shown}."
                    f" Please narrow the pattern or specify a subdirectory and retry.")
        return out

    # ── read_file ──

    def h_read_file(self, args: dict) -> str:
        from pi_agent._tools_impl import read_file
        filepath = args["filepath"]
        data_exts = (".csv", ".tsv", ".npz", ".npy", ".parquet", ".pkl", ".pickle")
        is_data_file = any(filepath.lower().endswith(ext) for ext in data_exts)
        if is_data_file:
            self._print(f"  🛑 Data file protection: {filepath} only showing first 500 chars")
            result = read_file(filepath, max_chars=500)
            content = result.get("content", "(empty)")
            return content + "\n\n⚠️ [System forced truncation] Above only shows the file header. To analyze data, use write_file to write a script + run_shell to execute."
        result = read_file(filepath, max_chars=250000)
        if result.get("structure"):
            return json.dumps(result["structure"], ensure_ascii=False)
        return result.get("content", "(empty)")

    # ── write_file ──

    def h_write_file(self, args: dict) -> str:
        from pi_agent._tools_impl import write_file

        if "_json_error" in args:
            raw = args.get('_raw_preview', '')
            raw_len = args.get('_raw_len', len(raw))
            filepath_hint = raw.split('"filepath": "')[1].split('"')[0] if '"filepath": "' in raw else "?"
            if filepath_hint.lower().endswith(".md"):
                hint = f"Memory file is {raw_len} chars long, JSON escaping failed. Please condense the content."
            else:
                hint = f"Code file is {raw_len} chars long, JSON escaping failed. Please condense or split into multiple files."
            return (
                f"❌ JSON exploded: {hint}\n"
                f"Target file: {filepath_hint}\n"
                f"⚠️ Do NOT retry with content of the same length! It will NOT succeed!"
            )

        filepath = args.get("filepath", "")
        content = args.get("content", "")
        if not filepath:
            return "❌ write_file failed: missing filepath parameter."
        if not content:
            return "❌ write_file failed: missing content parameter (code content is empty)."

        # ⛔ Feedback files are read-only
        if "workspace/feedback/" in filepath.lower().replace("\\", "/"):
            return "⛔ Modifying feedback files is forbidden! Feedback is manually maintained by the user."

        # MEMORY.md pre-write backup
        mem_backup = None
        if filepath.lower().endswith("memory.md") and os.path.exists(filepath):
            try:
                mem_backup = open(filepath, "r", encoding="utf-8").read()
            except Exception:
                pass

        result = write_file(filepath, content, args.get("mode", "overwrite"))
        ftype = "py" if filepath.lower().endswith(".py") else "file"
        msg = f"✅ write_{ftype}: {filepath} ({len(content)} chars)"

# MEMORY.md auto-repair
        if filepath.lower().endswith("memory.md"):
            try:
                mem_dir = os.path.dirname(os.path.abspath(filepath))
                actual_files = sorted(
                    f for f in os.listdir(mem_dir)
                    if f.endswith(".md") and f != "MEMORY.md"
                )
                links_in_content = re.findall(r'\[([^\]]+)\]\(([^)]+\.md)\)', content)
                linked_files = {l[1] for l in links_in_content}
                missing = [f for f in linked_files if f not in actual_files]
                unlisted = [f for f in actual_files if f not in linked_files]
                if missing or unlisted:
                    rebuilt = [f"# Agent Experiment Memory — {self.task_type}\n"]
                    for fn in actual_files:
                        desc = None
                        for label, link in links_in_content:
                            if link == fn:
                                desc = label; break
                        if not desc:
                            desc = fn.replace(".md", "")
                        rebuilt.append(f"- [{desc}]({fn})")
                    corrected = "\n".join(rebuilt) + "\n"
                    write_file(filepath, corrected, "overwrite")
                    msg += (
                        f"\n\n⚠️ MEMORY.md auto-repair:\n"
                        + (f"  Wrong links (removed): {', '.join(missing)}\n" if missing else "")
                        + (f"  Missing files (added): {', '.join(unlisted)}" if unlisted else "")
                    )
            except Exception:
                pass

        # MEMORY.md content protection
        if filepath.lower().endswith("memory.md") and mem_backup is not None:
            try:
                new_content = open(filepath, "r", encoding="utf-8").read()
                if len(new_content.strip().split("\n")) < 5 and len(mem_backup.strip().split("\n")) >= 5:
                    open(filepath, "w", encoding="utf-8").write(mem_backup)
                    msg += "\n\n🛡️ MEMORY.md protection: Agent wrote too little, backup restored."
                else:
                    import shutil
                    shutil.copy2(filepath, filepath + ".bak")
            except Exception:
                pass

        return msg

    # ── edit_file ──

    def h_edit_file(self, args: dict) -> str:
        raw_path = args["file_path"]
        if not os.path.isabs(raw_path):
            if raw_path.startswith(("workspace/", "workspace\\", "predictors/", "predictors\\")):
                filepath = os.path.abspath(raw_path)
            else:
                filepath = os.path.abspath(os.path.join("workspace", raw_path))
        else:
            filepath = os.path.abspath(raw_path)

        # Block edits to core infrastructure
        blocked_prefixes = [
            os.path.abspath("agent"), os.path.abspath("pi_agent"),
            os.path.abspath("main.py"), os.path.abspath("utils"),
        ]
        for blocked in blocked_prefixes:
            if filepath.startswith(blocked):
                return f"⛔ Modifying core infrastructure is forbidden! {filepath} is in a protected area."

        if not os.path.exists(filepath):
            return f"❌ File does not exist: {filepath}"

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        patches = args.get("patches")
        if patches:
            patch_list = [(p["old_string"], p["new_string"]) for p in patches]
        elif args.get("old_string"):
            patch_list = [(args["old_string"], args["new_string"])]
        else:
            return "❌ edit_file failed: old_string+new_string or patches parameter required."

        replace_all = args.get("replace_all", False)
        total_replaced = 0

        for old_str, new_str in patch_list:
            count = content.count(old_str)
            if count == 0:
                return (f"❌ old_string not found in file! Use read_file to verify the file contents."
                        f"Note: old_string must match exactly (including indentation and whitespace).\n"
                        f"Preview of unfound content: {old_str[:100]}")
            if not replace_all and count > 1:
                return (f"❌ old_string appeared {count} times in the file!"
                        f"Use replace_all=true to replace all, or provide more specific context.\n"
                        f"Match text preview: {old_str[:100]}")
            content = content.replace(old_str, new_str) if replace_all else content.replace(old_str, new_str, 1)
            total_replaced += count if replace_all else 1

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)

        detail = f"{len(patch_list)} groups, {total_replaced} occurrences" if patches else f"{total_replaced} occurrences"
        return f"✅ Replacement successful ({detail})"

    # ── run_shell ──

    def h_run_shell(self, args: dict) -> str:
        from pi_agent._tools_impl import run_shell
        cmd = args["command"]

        if "MEMORY.md" in cmd and any(op in cmd for op in ("rm ", "> ", "truncate", "/dev/null")):
            return "🛡️ Deleting/clearing MEMORY.md is forbidden!"

        # Auto-correct common path mistakes
        for wrong in ["/home/user/workspace", "/home/user/", "~/workspace"]:
            cmd = cmd.replace(wrong + "/", "").replace("cd " + wrong + " && ", "").replace(wrong, "")
        if re.search(r'\bcd\s+workspace/?\s*&&', cmd):
            cmd = re.sub(r'cd\s+workspace/?\s*&&\s*', '', cmd)

        result = run_shell(cmd, timeout=3600,
                          env_vars={"AGENT_BUDGET_REMAINING": "-1"})
        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()

        out = ""
        if stdout and stderr:
            out += f"[stdout]\n{stdout}\n\n[stderr]\n{stderr}"
        else:
            out += stdout or stderr or "(no output)"
        return out

    # ── start_shell ──

    def h_start_shell(self, args: dict) -> str:
        from pi_agent._tools_impl import start_shell
        command = args.get("command", "")
        timeout = args.get("timeout", 3600)
        result = start_shell(command, timeout=timeout)
        if result.get("success"):
            return (f"Process {result['pid']} started (status: {result['status']})\n"
                    f"Command: {result['command']}\n"
                    f"Hint: use check_shell({result['pid']}) to monitor output")
        return f"Start failed: {result.get('error', 'unknown')}"

    # ── check_shell ──

    def h_check_shell(self, args: dict) -> str:
        from pi_agent._tools_impl import check_shell
        pid = args.get("pid", -1)
        result = check_shell(pid)
        if not result.get("success"):
            return (
                f"check_shell failed: {result.get('error', 'unknown')}\n\n"
                f"⚠️ This process no longer exists! Do NOT check_shell({pid}) again."
                f"Please do something else: check output directories, read logs to analyze errors,"
                f"fix code and start_shell to retrain, or move on to other experiments."
            )
        out = f"Process {pid}: {result['status']} | Elapsed {result['elapsed']}s"
        if result['status'] == 'loading':
            out += "\n(Data loading phase; no output is normal, please wait patiently, do NOT kill)"
        if result.get("return_code") is not None:
            out += f" | return_code={result['return_code']}"
        if result.get("warning"):
            out += f"\n⚠️ {result['warning']}"
        if result.get("new_output"):
            limit = 4000 if result['status'] in ('error', 'completed') else 2000
            output_text = result['new_output']
            if len(output_text) > limit:
                half = limit // 2
                output_text = output_text[:half] + f"\n... [{len(output_text) - limit} chars omitted] ...\n" + output_text[-half:]
            out += f"\n--- New output ---\n{output_text}"
        if result.get("stderr"):
            limit = 3000 if result['status'] in ('error', 'completed') else 1000
            err_text = result['stderr']
            if len(err_text) > limit:
                err_text = err_text[-limit:]
            out += f"\n--- stderr ---\n{err_text}"
        return out

    # ── kill_shell ──

    def h_kill_shell(self, args: dict) -> str:
        from pi_agent._tools_impl import kill_shell
        pid = args.get("pid", -1)
        result = kill_shell(pid)
        if result.get("success"):
            out = f"Process {pid} terminated | Ran {result['elapsed']}s"
            if result.get("final_stderr"):
                out += f"\n--- Final stderr ---\n{result['final_stderr'][-2000:]}"
            return out
        return f"Termination failed: {result.get('error', 'unknown')}"

    # ── stop ──

    def h_stop(self, args: dict) -> str:
        if self._on_stop:
            self._on_stop()
        tag = "rec" if self.task_type == "recommendation" else "cls"
        memory_files = list(self.memory_dir.glob(f"{tag}-*.md"))
        feedback_path = Path(f"workspace/feedback/{self.task_type}.md")
        has_feedback = feedback_path.exists()
        msg = f"Stop signal received. {len(memory_files)} memory files."
        if has_feedback:
            feedback_content = feedback_path.read_text(encoding="utf-8", errors="replace")
            if any(line.strip().startswith("|") and "0." in line for line in feedback_content.split("\n")):
                msg += (
                    f"\n\n📋 Detected historical scores in online feedback file. On next run, this feedback will be read in the first round.\n"
                    f"P4 iron rule: after reading feedback, must write diagnostic hypotheses + validation experiments."
                )
        return msg + " Finalizing..."


def build_tool_manager(task_type: str, bench: str, memory_dir: Path,
                       print_fn: Callable, event_bus: EventBus = None) -> Tuple[ToolManager, ToolHandlers]:
    """
    Factory: create a fully-registered ToolManager with all 9 tools.

    Returns (manager, handlers) — manager for execution, handlers for Agent to inject callbacks.
    """
    handlers = ToolHandlers(task_type=task_type, bench=bench, memory_dir=memory_dir, print_fn=print_fn)
    manager = ToolManager(event_bus=event_bus, print_fn=print_fn)

    # Register all 10 tools
    manager.register("think", handlers.h_think)
    manager.register("list_files", handlers.h_list_files)
    manager.register("read_file", handlers.h_read_file)
    manager.register("write_file", handlers.h_write_file)
    manager.register("edit_file", handlers.h_edit_file)
    manager.register("run_shell", handlers.h_run_shell)
    manager.register("start_shell", handlers.h_start_shell)
    manager.register("check_shell", handlers.h_check_shell)
    manager.register("kill_shell", handlers.h_kill_shell)
    manager.register("stop", handlers.h_stop)

    return manager, handlers
