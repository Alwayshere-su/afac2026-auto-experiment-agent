"""全局配置常量"""
import os
from pathlib import Path

# 比赛约束
TOTAL_BUDGET_SECONDS = 7200  # 2 小时
SAFETY_MARGIN_SECONDS = 300  # 5 分钟安全余量

# ── API Key ──
# 在项目根目录创建 .api_key 文件，写入 DashScope API Key

def _resolve_api_key() -> str:
    key_file = Path(__file__).resolve().parent.parent / ".api_key"
    if key_file.exists():
        return key_file.read_text().strip()
    return ""

QWEN_API_KEY = _resolve_api_key()

# Qwen (DashScope)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-max")
QWEN_MAX_TOKENS = 16000
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/api/v1"

# Agent loop is purely time-driven — no round limit.
MAX_ROUNDS = 999
