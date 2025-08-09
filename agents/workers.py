# agents/workers.py
from agents.base_agent import BaseAgent
from tools.web_search import perform_search
from utils.helpers import format_search_results_for_llm

class ResearcherAgent(BaseAgent):
    """Выполняет базовое исследование: генерирует запросы, ищет, извлекает факты."""
    def execute(self, task: dict, model_name: str) -> dict:
        print(f"   [ResearcherAgent] -> Выполняю задачу '{task['task_id']}' на модели {model_name}...")
        try:
            # Шаг 1: Генерация поисковых запросов
            prompt_queries = f"Твоя роль: Ассистент-исследователь. Твоя задача: '{task['description']}'. Сгенерируй 3-4 точных поисковых запроса для этой задачи. Верни их как JSON-список строк."
            response_queries = self.llm_client.invoke(model_name, prompt_queries)
            queries = eval(response_queries.content)

            # Шаг 2: Поиск
            search_results_text = ""
            for q in queries:
                results = perform_search(q)
                search_results_text += format_search_results_for_llm(results) + "\n\n"

            # Шаг 3: Извлечение фактов
            prompt_claims = f"Твоя роль: Аналитик. Проанализируй текст ниже и извлеки 3-5 ключевых фактов (claims) в формате JSON-списка. Текст для анализа:\n{search_results_text}"
            response_claims = self.llm_client.invoke(model_name, prompt_claims)
            
            print(f"   [ResearcherAgent] <- Задача '{task['task_id']}' выполнена.")
            return {"status": "SUCCESS", "data": response_claims.content}
        except Exception as e:
            return {"status": "FAILURE", "error": str(e)}

class ContrarianAgent(ResearcherAgent):
    """Наследует логику Researcher, но с другим системным промптом (который будет вставлен в PromptEngine)."""
    pass

class QualityAssessorAgent(BaseAgent):
    """Оценивает качество фактов."""
    def execute(self, facts_to_assess: list, model_name: str) -> dict:
        print(f"   [QualityAssessorAgent] -> Оцениваю {len(facts_to_assess)} фактов на модели {model_name}...")
        # Логика оценки...
        return {"status": "SUCCESS", "data": {"good": facts_to_assess, "fixable": []}}

class FixerAgent(BaseAgent):
    """Исправляет некачественные факты."""
    def execute(self, facts_to_fix: list, model_name: str) -> dict:
        print(f"   [FixerAgent] -> Исправляю {len(facts_to_fix)} фактов на модели {model_name}...")
        # Логика исправления...
        return {"status": "SUCCESS", "data": facts_to_fix}

class AnalystAgent(BaseAgent):
    """Синтезирует инсайты из Базы Знаний."""
    def execute(self, knowledge_base: dict, model_name: str) -> dict:
        print(f"   [AnalystAgent] -> Синтезирую инсайты на модели {model_name}...")
        # Логика анализа...
        return {"status": "SUCCESS", "data": {"insights": "ключевые выводы", "gaps": []}}

class ReportWriterAgent(BaseAgent):
    """Пишет отчет по структурированным данным."""
    def execute(self, analysis_data: dict, model_name: str) -> dict:
        print(f"   [ReportWriterAgent] -> Пишу отчет на модели {model_name}...")
        # Логика написания...
        return {"status": "SUCCESS", "data": "# Финальный Отчет"}