"""BudgetTracker 的纯逻辑测试（不依赖 torch / 比赛数据）。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.budget_tracker import BudgetTracker  # noqa: E402


class BudgetTrackerTests(unittest.TestCase):
    def make_tracker(self, budget: int = 100, margin: int = 10) -> BudgetTracker:
        return BudgetTracker(total_budget=budget, safety_margin=margin)

    def test_initial_state(self) -> None:
        tracker = self.make_tracker()
        self.assertEqual(tracker.elapsed(), 0.0)
        self.assertEqual(tracker.remaining(), 100.0)
        self.assertEqual(tracker.effective_remaining(), 90.0)
        self.assertFalse(tracker.is_exhausted())
        self.assertFalse(tracker.must_stop_now())

    def test_elapsed_follows_clock(self) -> None:
        tracker = self.make_tracker()
        tracker.start()
        tracker.start_time -= 30.0  # 模拟已运行 30 秒
        self.assertAlmostEqual(tracker.elapsed(), 30.0, delta=1.0)
        self.assertAlmostEqual(tracker.remaining(), 70.0, delta=1.0)
        self.assertAlmostEqual(tracker.budget_used_pct(), 30.0, delta=1.0)

    def test_pause_excludes_paused_time(self) -> None:
        tracker = self.make_tracker()
        tracker.start()
        tracker.start_time -= 100.0
        tracker.pause()
        tracker._pause_start -= 40.0  # 已暂停 40 秒
        self.assertAlmostEqual(tracker.elapsed(), 60.0, delta=1.0)

        tracker.resume()
        self.assertAlmostEqual(tracker.elapsed(), 60.0, delta=1.0)
        self.assertFalse(tracker._paused)

    def test_remaining_never_negative(self) -> None:
        tracker = self.make_tracker(budget=10, margin=2)
        tracker.start()
        tracker.start_time -= 1_000.0
        self.assertEqual(tracker.remaining(), 0.0)
        self.assertEqual(tracker.effective_remaining(), 0.0)
        self.assertTrue(tracker.is_exhausted())
        self.assertTrue(tracker.must_stop_now())

    def test_safety_margin_boundary(self) -> None:
        tracker = self.make_tracker(budget=10, margin=3)
        tracker.start()
        tracker.start_time -= 8.0  # 剩 2 秒，已进入安全余量
        self.assertFalse(tracker.is_exhausted())
        self.assertTrue(tracker.must_stop_now())


if __name__ == "__main__":
    unittest.main()
