"""
会话管理 — Pi-Agent Layer 2
============================
支持对话状态的保存、恢复和删除操作。

相比原始 CheckpointManager，保存了恢复所需的全部状态：
消息历史、预算消耗、迭代计数、轨迹日志，而不仅仅是消息列表。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class SessionManager:
    """
    会话持久化：保存/恢复对话 + Agent 状态。

    用法:
        sm = SessionManager("classification")
        sm.save(iteration=5, messages=[...], budget_elapsed=120.0,
                trajectory=[...], experiments_completed=3)
        data = sm.load()  # 返回 dict 或 None
    """

    def __init__(self, task_type: str, checkpoint_dir: str = "workspace"):
        tag = "cls" if task_type == "classification" else "rec"
        self._path = Path(checkpoint_dir) / f"checkpoint_{tag}.json"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── Save ──

    def save(self, iteration: int, messages: List[Dict], budget_elapsed: float,
             best_val: float = 0.0, summary: str = "",
             trajectory: List[Dict] = None,
             experiments_completed: int = 0) -> bool:
        """Save full session state to disk."""
        try:
            data = {
                "iteration": iteration,
                "messages": messages,
                "budget_elapsed": budget_elapsed,
                "best_val": best_val,
                "summary": summary,
                "trajectory": trajectory or [],
                "experiments_completed": experiments_completed,
                "timestamp": datetime.now().isoformat(),
            }
            self._path.write_text(json.dumps(data, ensure_ascii=False, default=str), encoding="utf-8")
            return True
        except Exception:
            return False

    # ── Load ──

    def load(self) -> Optional[Dict]:
        """Load saved session state. Returns None if no checkpoint exists."""
        if not self._path.exists():
            return None
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def exists(self) -> bool:
        return self._path.exists()

    # ── Delete ──

    def delete(self) -> bool:
        """Delete the checkpoint file (called on clean completion)."""
        try:
            if self._path.exists():
                self._path.unlink()
            return True
        except OSError:
            return False


