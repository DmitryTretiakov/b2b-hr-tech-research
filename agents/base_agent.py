# agents/base_agent.py
from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager

class BaseAgent:
    def __init__(self, llm_client: LLMClient, budget_manager: APIBudgetManager):
        self.llm_client = llm_client
        self.budget_manager = budget_manager
        print(f"-> Агент '{self.__class__.__name__}' инициализирован.")