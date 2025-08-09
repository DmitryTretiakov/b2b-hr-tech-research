# agents/supervisor.py
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from pydantic import BaseModel, Field
from typing import List, Dict

class GraphPlan(BaseModel):
    """Pydantic-модель для описания плана графа."""
    tasks: List[Dict] = Field(description="Список всех задач, которые нужно выполнить.")
    initial_model_assignments: Dict[str, str] = Field(description="Словарь {task_id: model_name} с начальным распределением моделей.")
    # Примечание: ребра (edges) будут управляться логикой оркестратора

class SupervisorAgent(BaseAgent):
    """
    Генерирует первоначальный план для графа, соблюдая иерархию ресурсов.
    Использует модель Уровня 4 (Pro).
    """
    def create_initial_plan(self, main_goal: str) -> Dict:
        model_name = "gemini-2.5-pro"
        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Главный Архитектор AI-систем. Твоя задача - создать первоначальный, ресурсоэффективный план исследования в виде графа задач.

**ГЛАВНАЯ ЦЕЛЬ ИССЛЕДОВАНИЯ:**
{main_goal}

**ФИЛОСОФИЯ РАСПРЕДЕЛЕНИЯ РЕСУРСОВ (СТРОГО СОБЛЮДАТЬ):**
- **Уровень 2 (Gemma 3 / Flash-Lite):** Для 90% рутинных задач (Researcher, QualityAssessor, Fixer, ReportWriter).
- **Уровень 3 (Flash):** Для сложных аналитических задач (Analyst, KnowledgeJanitor).
- **Уровень 4 (Pro):** Не назначать. Этот уровень зарезервирован для тебя и мета-агентов.

**ТВОЯ ЗАДАЧА:**
1. Декомпозируй главную цель на логические этапы (например, "Анализ конкурентов", "Оценка рынка", "Техническая экспертиза").
2. Для каждого этапа создай набор задач. Обязательно включай состязательные пары (`Researcher` и `Contrarian`).
3. Для КАЖДОЙ задачи в поле `initial_model_assignments` назначь **самую дешевую подходящую модель** из Уровня 2 (например, 'gemma-3').
4. Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
"""
        print("   [SupervisorAgent] -> Генерирую ресурсоэффективный план графа...")
        plan_data = invoke_llm_for_json_with_retry(
            main_llm=self.llm_client._get_model_instance(model_name),
            sanitizer_llm=self.llm_client._get_model_instance("gemini-2.5-flash"), # Санитайзер уровнем ниже
            prompt=prompt,
            pydantic_schema=GraphPlan,
            budget_manager=self.budget_manager,
            model_name_for_budget=model_name
        )
        print("   [SupervisorAgent] <- План графа успешно сгенерирован.")
        return plan_data