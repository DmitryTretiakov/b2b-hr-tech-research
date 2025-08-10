# agents/base_agent.py
from __future__ import annotations
from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager
if 'ToolRegistry' not in locals():
    from core.tool_registry import ToolRegistry

class BaseAgent:
    """
    Базовый класс для всех агентов в системе.
    Теперь включает опциональную поддержку реестра инструментов.
    """
    def __init__(self, llm_client: LLMClient, budget_manager: APIBudgetManager, tool_registry: ToolRegistry = None):
        self.llm_client = llm_client
        self.budget_manager = budget_manager
        self.tool_registry = tool_registry  # Может быть None для агентов, которые не используют инструменты
        print(f"-> Агент '{self.__class__.__name__}' инициализирован.")