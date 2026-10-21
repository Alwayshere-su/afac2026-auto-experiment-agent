"""预算追踪器：管理 2 小时时钟时间预算"""
import time
from typing import Optional
from utils.config import TOTAL_BUDGET_SECONDS, SAFETY_MARGIN_SECONDS


class BudgetTracker:
    """追踪实验消耗的时钟时间。

    支持暂停/恢复以处理非计算时间（如 API 等待）。
    """

    def __init__(self, total_budget: int = TOTAL_BUDGET_SECONDS,
                 safety_margin: int = SAFETY_MARGIN_SECONDS):
        self.total_budget = total_budget
        self.safety_margin = safety_margin
        self.start_time: Optional[float] = None
        self._paused = False
        self._pause_start: Optional[float] = None
        self._paused_duration = 0.0

    def start(self):
        """Start the budget timer"""
        self.start_time = time.time()

    def elapsed(self) -> float:
        """Elapsed time in seconds, excluding paused periods"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time - self._paused_duration
        if self._paused and self._pause_start is not None:
            elapsed -= (time.time() - self._pause_start)
        return elapsed

    def remaining(self) -> float:
        """Remaining budget in seconds"""
        return max(0.0, self.total_budget - self.elapsed())

    def effective_remaining(self) -> float:
        """Remaining budget minus safety margin"""
        return max(0.0, self.remaining() - self.safety_margin)

    def is_exhausted(self) -> bool:
        """True if budget is fully consumed"""
        return self.remaining() <= 0

    def must_stop_now(self) -> bool:
        """True if we must stop considering safety margin"""
        return self.effective_remaining() <= 0

    def pause(self):
        """Pause the timer (for API calls, etc.)"""
        if not self._paused:
            self._paused = True
            self._pause_start = time.time()

    def resume(self):
        """Resume the timer"""
        if self._paused and self._pause_start is not None:
            self._paused_duration += time.time() - self._pause_start
            self._paused = False
            self._pause_start = None

    def budget_used_pct(self) -> float:
        """Percentage of budget used"""
        return self.elapsed() / self.total_budget * 100
