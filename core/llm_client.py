# core/llm_client.py
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager

class LLMClient:
    """
    Централизованный клиент для взаимодействия с LLM.
    Инкапсулирует иерархию моделей и управление бюджетом.
    """
    def __init__(self, budget_manager: APIBudgetManager):
        self.budget_manager = budget_manager
        
        # Инициализация моделей согласно иерархии из ТЗ
        self._models = {
            # Уровень 4
            "gemini-2.5-pro": ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3),
            # Уровень 3
            "gemini-2.5-flash": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1),
            # Уровень 2
            "gemini-2.5-flash-lite": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.1),
            "gemma-3": ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", temperature=0.0),
            # Уровень 1 (пример, можно расширить)
            "gemma-3n": ChatGoogleGenerativeAI(model="models/gemma-3-12b-it", temperature=0.0),
        }
        print("-> LLMClient инициализирован с утвержденной иерархией моделей.")

    def _get_model_instance(self, model_name: str) -> ChatGoogleGenerativeAI:
        instance = self._models.get(model_name)
        if not instance:
            raise ValueError(f"Модель '{model_name}' не найдена в клиенте. Проверьте конфигурацию.")
        return instance

    def invoke(self, model_name: str, prompt: str):
        """
        Выполняет вызов к указанной модели с контролем бюджета.
        """
        if not self.budget_manager.can_i_spend(model_name):
            raise ConnectionError(f"Дневной лимит для модели {model_name} исчерпан.")
            
        print(f"   [LLMClient] -> Вызов модели Уровня '{self.get_level(model_name)}': {model_name}")
        
        instance = self._get_model_instance(model_name)
        response = instance.invoke(prompt)
        
        self.budget_manager.record_spend(model_name)
        return response

    def get_level(self, model_name: str) -> int:
        """Возвращает уровень иерархии для модели."""
        if "pro" in model_name: return 4
        if "2.5-flash" in model_name and "lite" not in model_name: return 3
        if "flash-lite" in model_name or "gemma-3" in model_name: return 2
        if "gemma-3n" in model_name: return 1
        return 0