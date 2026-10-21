"""
LLM 调用抽象层 — Pi-Agent Layer 1
===================================
提供统一的 LLM 调用接口，封装 Qwen (DashScope) API。

特性：
  - Tool call 参数 JSON 自动修复（LLM 输出格式错误时）
  - API 调用期间心跳动画（防止看起来卡死）
  - 指数退避重试（最多 5 次）
  - 纯文本推理模式（think 工具专用，不触发 tool call）
  - 重复 tool call 检测与自动修复
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Tool definitions ──

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Deep reasoning tool. Call this when you need to analyze complex situations, form hypotheses, or plan strategy. The system will give you space to reason without the pressure of selecting other tools.\n\n"
                           "Use think when: you've gathered enough information and need to form a hypothesis before experimenting; you've received unexpected results and need to understand why; you're at a decision point with multiple options.\n\n"
                           "Do NOT use think for: simple file reads, monitoring training, or routine operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "What to think about, e.g. 'Analyze whether degree-weighted multi-K GCN can beat the current best score of 0.40915' or 'Why did this experiment fail?'"
                    }
                },
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Recursively list all files in a directory. Used to discover available data, code, log, and output files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path to list, e.g. workspace or workspace/outputs",
                        "default": "workspace"
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Glob filter pattern, e.g. *.csv or *.json",
                        "default": "**/*"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read any project file. Can read memory/logs/data, previously generated code, and project source under agent/ predictors/. JSON files display a structural summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "File path, e.g. workspace/outputs/_research_knowledge_rec.json"
                    }
                },
                "required": ["filepath"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a file. Supports .md (memory) and .py (code), no content length limit. Special characters in content (quotes, newlines, backslashes) are auto-escaped — pass raw code directly, no manual handling needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {
                        "type": "string",
                        "description": "Target file path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write. Quotes/newlines/backslashes are auto-escaped, pass raw Python code directly"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["overwrite", "append"],
                        "description": "Write mode",
                        "default": "overwrite"
                    }
                },
                "required": ["filepath", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Edit a file via exact string replacement. Supports two modes:\n\n"
                           "Single replacement: pass old_string + new_string, replaces first occurrence (add replace_all=true to replace all)\n"
                           "Batch replacement: pass patches=[{\"old_string\":\"...\",\"new_string\":\"...\"}, ...] to change multiple different locations at once",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path of the file to modify"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Text to replace, must match exactly (including indentation). Mutually exclusive with patches."
                    },
                    "new_string": {
                        "type": "string",
                        "description": "Replacement text. Mutually exclusive with patches."
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Whether to replace all matches (default false, only replaces first occurrence)",
                        "default": False
                    },
                    "patches": {
                        "type": "array",
                        "description": "Batch replacement list, each entry has old_string and new_string. Mutually exclusive with old_string/new_string.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string", "description": "Text to replace"},
                                "new_string": {"type": "string", "description": "Replacement text"}
                            },
                            "required": ["old_string", "new_string"]
                        }
                    }
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command in the workspace sandbox (3600s timeout). python automatically points to project .venv Python; torch/numpy/pandas/scipy/sklearn/dashscope etc. are all installed.\n\n"
                           "To run a training script: python workspace/code/xxx.py (script must be self-contained with explicit imports of all libraries; load data yourself). "
                           "Write code with write_file first, then execute with run_shell.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_shell",
            "description": "Non-blocking launch of a long-running command (e.g. training script). Returns process ID immediately, program runs in background. "
                           "Agent monitors output every few seconds with check_shell; kill_shell to terminate early if issues detected. "
                           "Do NOT use run_shell for training scripts — it blocks until the script finishes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute, e.g. 'python workspace/code/train.py'"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Timeout in seconds, default 3600",
                        "default": 3600
                    }
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_shell",
            "description": "Check background process status and new output. Each call auto-blocks for up to 90 seconds, returning early if output appears or the process ends. "
                           "Used for real-time monitoring of training scripts — view loss curves, detect Traceback/CUDA errors.\n\n"
                           "**status meanings**: loading=loading data (no first line of output yet, wait patiently); running=normal operation; completed=finished; "
                           "error=crashed; stuck=had output but no new output for >2 min (genuinely stuck or zombie process).\n\n"
                           "do NOT kill loading status! Data loading/co-visit matrix construction may have no output for 1-2 minutes, this is normal. "
                           "Only kill on stuck or error. For zombie processes (files saved but process won't exit), kill then use list_files to check output directly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "integer",
                        "description": "Process ID returned by start_shell"
                    }
                },
                "required": ["pid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kill_shell",
            "description": "Terminate a background process. Use when training issues are detected (error/stuck/loss not decreasing), then edit_file to fix code and start_shell to restart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pid": {
                        "type": "integer",
                        "description": "Process ID to terminate"
                    }
                },
                "required": ["pid"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Stop the experiment. Call after generating CSV + writing memory files. No parameters, just signals completion.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
]


# ═══════════════════════════════════════════════════════════════
# Provider Registry
# ═══════════════════════════════════════════════════════════════

class Provider:
    """Abstract base for LLM providers."""

    def __init__(self, name: str, model: str, max_tokens: int):
        self.name = name
        self.model = model
        self.max_tokens = max_tokens

    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             temperature: float = 0.1) -> Tuple[str, str, List[Dict]]:
        """
        Returns (content, reasoning_content, tool_calls_list).
        Each tool call in the list is: {"id": ..., "type": "function", "function": {"name": ..., "arguments": ...}}
        """
        raise NotImplementedError



class QwenProvider(Provider):
    """DashScope native Qwen API."""

    def __init__(self):
        from utils.config import QWEN_MODEL, QWEN_MAX_TOKENS, QWEN_BASE_URL, QWEN_API_KEY
        super().__init__("qwen", QWEN_MODEL, QWEN_MAX_TOKENS)
        self.base_url = QWEN_BASE_URL
        self.api_key = QWEN_API_KEY

    def chat(self, messages: List[Dict], tools: List[Dict] = None,
             temperature: float = 0.1) -> Tuple[str, str, List[Dict]]:
        import dashscope
        dashscope.base_http_api_url = self.base_url
        kwargs = dict(
            api_key=self.api_key, model=self.model,
            messages=messages, max_tokens=self.max_tokens,
            result_format="message", enable_thinking=True,
        )
        if tools:
            kwargs["tools"] = tools
        resp = dashscope.Generation.call(**kwargs)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Error code: {resp.status_code} - "
                f"{{'error': {{'message': '{resp.message}', 'code': '{resp.code}'}}}}"
            )
        msg = resp.output.choices[0].message
        reasoning = msg.get('reasoning_content') or ''
        content = msg.get('content') or ''
        if not content and reasoning:
            content = reasoning
        raw_tcs = msg.get('tool_calls') or []
        tcs = []
        for tc in raw_tcs:
            if isinstance(tc, dict):
                fn = tc.get("function", {})
                tcs.append({
                    "id": tc.get("id", ""), "type": tc.get("type", "function"),
                    "function": {"name": fn.get("name", ""), "arguments": fn.get("arguments", "")}
                })
            else:
                tcs.append({
                    "id": getattr(tc, 'id', ''), "type": getattr(tc, 'type', 'function'),
                    "function": {"name": getattr(tc.function, 'name', ''), "arguments": getattr(tc.function, 'arguments', '')}
                })
        return content, reasoning, tcs


# ═══════════════════════════════════════════════════════════════
# LLM Client (Unified Interface)
# ═══════════════════════════════════════════════════════════════

class LLMClient:
    """
    Unified LLM client with multi-provider support and auto-failover.

    Usage:
        client = LLMClient()
        content, reasoning, tool_calls = client.call_with_tools(messages)
    """

    def __init__(self, primary: str = None, print_fn: Callable = None):
        self._print = print_fn or (lambda x: None)
        self._providers: Dict[str, Provider] = {}
        self._active: Optional[str] = None

        try:
            qwen = QwenProvider()
            if qwen.api_key:
                self._providers["qwen"] = qwen
                self._active = "qwen"
                self._print(f"[LLM] Qwen initialized (model: {qwen.model})")
        except Exception as e:
            self._print(f"[LLM] Qwen init failed: {e}")

        if not self._providers:
            raise RuntimeError("No LLM provider available! Create .api_key file in project root with your DashScope API Key.")

        self._print(f"[LLM] Active: qwen")

    @property
    def available(self) -> bool:
        return len(self._providers) > 0

    @property
    def active_provider_name(self) -> str:
        return self._active or "none"

    @property
    def _active_provider(self) -> Provider:
        return self._providers[self._active]

    # ── JSON Repair (same logic as original) ──

    @staticmethod
    def repair_json(raw: str) -> str:
        """Attempt to repair common LLM JSON errors."""
        if not raw or not isinstance(raw, str):
            return raw
        result = []
        in_string = False
        escape_next = False
        string_start = -1
        i = 0
        while i < len(raw):
            ch = raw[i]
            if escape_next:
                result.append(ch); escape_next = False; i += 1; continue
            if ch == '\\':
                result.append(ch); escape_next = True; i += 1; continue
            if ch == '"':
                if in_string:
                    j = i + 1
                    while j < len(raw) and raw[j] in ' \t\r\n':
                        j += 1
                    next_ch = raw[j] if j < len(raw) else ''
                    if j >= len(raw) or next_ch in ',}]':
                        result.append('"'); in_string = False
                    elif next_ch == ':':
                        if string_start >= 0 and (i - string_start) > 30:
                            result.append('\\"')
                        else:
                            result.append('"'); in_string = False
                    else:
                        result.append('\\"')
                else:
                    result.append('"'); in_string = True; string_start = i
                i += 1; continue
            if in_string:
                if ch == '\n': result.append('\\n')
                elif ch == '\r': result.append('\\r')
                elif ch == '\t': result.append('\\t')
                else: result.append(ch)
            else:
                result.append(ch)
            i += 1
        repaired = ''.join(result)
        if in_string:
            repaired += '"'
        import re
        repaired = re.sub(r',\s*([}\]])', r'\1', repaired)
        return repaired

    # ── Main API ──

    def call_with_tools(self, messages: List[Dict]) -> Tuple[Optional[str], Optional[str], Optional[List[Dict]]]:
        """
        Call LLM with tool definitions. Auto-retries + provider switch on failure.

        Returns (content, reasoning, tool_calls) — each can be None on total failure.
        """
        if not self._providers:
            return None, None, None

        null_streak = 0
        last_error = ""

        for attempt in range(5):
            _heartbeat_stop = False
            try:
                # Heartbeat spinner thread
                spinner = ["|", "/", "-", "\\"]

                def _heartbeat():
                    frame = 0
                    for _ in range(600):
                        if _heartbeat_stop: break
                        time.sleep(0.5)
                        if not _heartbeat_stop:
                            print(f"\r     {spinner[frame % 4]}", end="", flush=True)
                            frame += 1

                hb = threading.Thread(target=_heartbeat, daemon=True)
                hb.start()

                provider = self._active_provider
                content, reasoning, tcs = provider.chat(messages, tools=TOOL_DEFINITIONS)

                _heartbeat_stop = True; hb.join(timeout=1)

                if not content and reasoning:
                    content = reasoning

                null_streak = 0
                return content, reasoning, (tcs if tcs else None)

            except Exception as e:
                _heartbeat_stop = True
                last_error = str(e)
                null_streak += 1

                # Repetitive tool call detection (Qwen-specific)
                if "Repetitive tool calls" in last_error or "InvalidParameter" in last_error:
                    cut_at = None
                    for i in range(len(messages) - 1, -1, -1):
                        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
                            cut_at = i; break
                    if cut_at is not None and cut_at > 1:
                        removed = len(messages) - cut_at
                        del messages[cut_at:]
                        self._print(f"  [LLM] Repetitive call detected, removed last {removed} messages")
                    dead_pid_hint = ""
                    for i in range(len(messages) - 1, max(-1, len(messages) - 6), -1):
                        m = messages[i]
                        if m.get("role") == "tool" and "不存在" in str(m.get("content", "")):
                            import re as _re
                            mp = _re.search(r'进程\s*(\d+)', str(m.get("content", "")))
                            if mp:
                                dead_pid_hint = (
                                    f"(Process {mp.group(1)} is dead, do NOT check again. "
                                    f"Use list_files to see output, read logs to find bugs, fix code and retrain.)"
                                )
                            break
                    messages.append({
                        "role": "user",
                        "content": (
                            "[System] ⚠️ Consecutive repetitive calls detected, conversation history has been truncated."
                            "You MUST vary your actions — do NOT call the same tool with the same parameters again!"
                            + dead_pid_hint +
                            "Available actions: list_files to check output directories, read_file to read logs/memory/feedback,"
                            "write_file to write analysis or fix code, start_shell to launch new training,"
                            "read_file to read reference docs for new directions."
                        )
                    })

                # Auto-switch disabled (Qwen has no money)
                # if null_streak >= 2 and len(self._providers) > 1:
                #     ...

            if attempt < 4:
                delay = min(3 * (2 ** attempt), 60)
                self._print(f"  [LLM] Retry {attempt+1}/5 in {delay}s: {last_error}")
                time.sleep(delay)

        self._print(f"  ⚠️ All 5 API retries exhausted, auto-stop to save existing results")
        return None, None, None

    def think(self, messages: List[Dict], max_tokens: int = 800) -> str:
        """Text-only call for deep reasoning — no tools, just thinking.

        Used for the Think phase before the Act phase, to let the LLM
        reason deeply without the pressure of selecting tools.
        """
        if not self._providers:
            return ""
        try:
            provider = self._active_provider
            if provider_name == "qwen" and isinstance(provider, QwenProvider):
            # Qwen: convert system message to system_prompt param
                resp = provider.client.chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    max_tokens=min(max_tokens, provider.max_tokens),
                    timeout=60,
                )
                if resp and resp.choices:
                    msg = resp.choices[0].message
                    return (msg.content or getattr(msg, 'reasoning_content', '') or '')
            else:
                import dashscope
                resp = dashscope.Generation.call(
                    api_key=provider.api_key,
                    model=provider.model,
                    messages=messages,
                    max_tokens=min(max_tokens, provider.max_tokens),
                    result_format="message",
                    enable_thinking=True,
                )
                if resp.status_code == 200 and resp.output and resp.output.choices:
                    msg = resp.output.choices[0].message
                    return (msg.get('content') or msg.get('reasoning_content') or '')
        except Exception:
            pass
        return ""
