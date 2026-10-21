"""
Pi-Agent: 自主机器学习实验 Agent

三层架构：
  Layer 1: LLM 抽象层  (llm.py)      — 统一的 Qwen API 调用接口
  Layer 2: Agent 运行时  (agent.py)    — ReAct 循环，事件驱动，状态机
  Layer 3: CLI 入口     (main.py)     — 命令行接口

功能上与原始 ToolCallBrain 等价，但按 Pi-Agent 的模块化事件驱动架构重组。
"""

from pi_agent.agent import PiAgent
from pi_agent.main import main

__all__ = ["PiAgent", "main"]
