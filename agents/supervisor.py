# agents/supervisor.py
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import GraphPlan
from typing import List, Dict

class SupervisorAgent(BaseAgent):
    """
    Генерирует первоначальный и последующие планы для графа, соблюдая иерархию ресурсов.
    Использует модель Уровня 4 (Pro).
    """
    def create_initial_plan(self, main_goal: str) -> Dict:
        model_name = "gemini-2.5-pro"
        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Главный Архитектор AI-систем. Твоя задача - создать первоначальный, ресурсоэффективный план исследования в виде графа задач.

**ГЛАВНАЯ ЦЕЛЬ ИССЛЕДОВАНИЯ:**
{main_goal}

**ФИЛОСОФИЯ РАСПРЕДЕЛЕНИЯ РЕСУРСОВ (СТРОГО СОБЛЮДАТЬ):**
- **Уровень 2 (gemma-3 / gemini-2.5-flash-lite):** Для 90% рутинных задач (Researcher, Contrarian, QualityAssessor, Fixer, ReportWriter).
- **Уровень 3 (gemini-2.5-flash):** Для сложных аналитических задач (Analyst, KnowledgeJanitor).
- **Уровень 4 (gemini-2.5-pro):** Не назначать. Этот уровень зарезервирован для тебя и мета-агентов.

**ТВОЯ ЗАДАЧА:**
1. Декомпозируй главную цель на логические этапы (например, "Анализ конкурентов", "Оценка рынка", "Техническая экспертиза").
2. Для каждого этапа создай набор задач. Обязательно включай состязательные пары (`Researcher` и `Contrarian`).
3. Для КАЖДОЙ задачи в поле `initial_model_assignments` назначь **самую дешевую подходящую модель** из Уровня 2 (например, 'gemma-3').
4. Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
"""
        print("   [SupervisorAgent] -> Генерирую ресурсоэффективный план графа...")
        plan_data = invoke_llm_for_json_with_retry(
            llm_client=self.llm_client,
            model_name=model_name,
            sanitizer_model_name="gemini-2.5-flash",
            prompt=prompt,
            pydantic_schema=GraphPlan,
            budget_manager=self.budget_manager
        )
        print("   [SupervisorAgent] <- План графа успешно сгенерирован.")
        return plan_data if plan_data else {"tasks": [], "initial_model_assignments": {}}

    def create_next_phase_plan(self, analysis_summary: dict) -> Dict:
        """Генерирует план для следующей фазы на основе анализа предыдущей."""
        model_name = "gemini-2.5-pro"
        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Главный Архитектор AI-систем. Предыдущая фаза исследования завершена. Твоя задача - спланировать СЛЕДУЮЩУЮ фазу.

**СВОДКА ПРЕДЫДУЩЕЙ ФАЗЫ (от AnalystAgent):**
- **Ключевые Выводы:** {analysis_summary.get('key_insights', 'Нет данных')}
- **Обнаруженные Пробелы в Данных:** {analysis_summary.get('data_gaps', 'Нет данных')}

**ФИЛОСОФИЯ РАСПРЕДЕЛЕНИЯ РЕСУРСОВ (СТРОГО СОБЛЮДАТЬ):**
- **Уровень 2 (gemma-3 / gemini-2.5-flash-lite):** Для 90% рутинных задач.
- **Уровень 3 (gemini-2.5-flash):** Для сложных аналитических задач.

**ТВОЯ ЗАДАЧА:**
1.  Проанализируй выводы и, что более важно, **пробелы в данных**.
2.  Сгенерируй новый набор задач, нацеленных на **закрытие этих пробелов**.
3.  Если пробелов нет, сгенерируй задачи для более глубокого изучения ключевых выводов.
4.  Если и то, и другое не требуется, верни пустой список задач `[]`.
5.  Для КАЖДОЙ задачи назначь **самую дешевую подходящую модель** (например, 'gemma-3').
6.  Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
"""
        print("   [SupervisorAgent] -> Генерирую план для следующей фазы...")
        plan_data = invoke_llm_for_json_with_retry(
            llm_client=self.llm_client,
            model_name=model_name,
            sanitizer_model_name="gemini-2.5-flash",
            prompt=prompt,
            pydantic_schema=GraphPlan,
            budget_manager=self.budget_manager
        )
        if not plan_data:
            print("   [SupervisorAgent] <- Не удалось сгенерировать план следующей фазы.")
            return {"tasks": [], "initial_model_assignments": {}}
            
        print(f"   [SupervisorAgent] <- План следующей фазы сгенерирован ({len(plan_data.get('tasks', []))} задач).")
        return plan_data