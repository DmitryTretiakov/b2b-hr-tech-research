# agents/supervisor.py
import json
import uuid
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import StrategicPlan, AnalystReport, Task # Импортируем модели

class SupervisorAgent:
    """
    "Мозговой центр" системы. Отвечает только за планирование и стратегию.
    Использует самую мощную модель (Gemini 2.5 Pro).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print("-> SupervisorAgent (на базе Gemini 2.5 Pro) готов к работе.")

    def create_strategic_plan(self, world_model_context: dict) -> dict:
        """
        Генерирует первоначальный план, создавая состязательные пары задач.
        """
        print("   [Supervisor] Создаю первоначальный стратегический план с состязательными задачами...")

        prompt = f"""**ОБЩИЙ КОНТЕКСТ ПРОЕКТА:**
---
{json.dumps(world_model_context['static_context'], ensure_ascii=False, indent=2)}
---
**ТВОЯ РОЛЬ:** Ты - Главный Продуктовый Стратег. Твоя задача - создать комплексный, пошаговый план исследования.
**КЛЮЧЕВОЙ ПРИНЦИП (СОСТЯЗАТЕЛЬНОСТЬ):** Для каждой исследовательской цели ты ОБЯЗАН создать ДВЕ задачи:
1.  **ResearcherAgent:** Ищет подтверждения, позитивные данные, рыночные успехи.
2.  **ContrarianAgent:** Агрессивно ищет опровержения, критику, провалы, риски.

**ТВОЯ ЗАДАЧА:**
Проанализируй "ВХОДНОЙ БРИФ". Сгенерируй план из 3-4 логических фаз. Для каждой фазы создай 2-3 **пары** состязательных задач.
Для каждой пары задач используй одинаковый `pair_id`.

**ПРИМЕР ПАРЫ ЗАДАЧ:**
- "task_id": "task_001_research", "assignee": "ResearcherAgent", "description": "Найти 3-5 успешных конкурентов...", "pair_id": "pair_001"
- "task_id": "task_001_contra", "assignee": "ContrarianAgent", "description": "Найти 3-5 провалившихся конкурентов...", "pair_id": "pair_001"

Ты ОБЯЗАН вернуть результат в формате JSON.
"""
        plan = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=StrategicPlan,
            budget_manager=self.budget_manager
        )

        if plan and "phases" in plan:
            print("   [Supervisor] Первоначальный состязательный план успешно сгенерирован.")
            return plan
        else:
            print("!!! Supervisor: Не удалось сгенерировать план.")
            return {"main_goal_status": "FAILED", "phases": []}

    def reflect_and_update_plan(self, analyst_report: dict, current_plan: dict) -> dict:
        """
        Проводит рефлексию на основе концентрированного отчета от Аналитика.
        """
        print("   [Supervisor] Провожу рефлексию на основе отчета Аналитика...")

        prompt = f"""**ТВОЯ РОЛЬ:** Ассистент-планировщик.
**ТВОЯ ЗАДАЧА:** Тебе предоставлен отчет от старшего аналитика и текущий план. Твоя задача - обновить план в соответствии с выводами.

**СТРУКТУРИРОВАННЫЙ ОТЧЕТ АНАЛИТИКА:**
---
{json.dumps(analyst_report, ensure_ascii=False, indent=2)}
---

**ТЕКУЩИЙ ПЛАН:**
---
{json.dumps(current_plan, ensure_ascii=False, indent=2)}
---

**ИНСТРУКЦИИ ПО ОБНОВЛЕНИЮ:**
1.  Заверши текущую активную фазу (статус "COMPLETED").
2.  Активируй следующую фазу (статус "IN_PROGRESS").
3.  Если аналитик обнаружил пробелы в данных (`data_gaps`), создай новые **пары состязательных задач** (для ResearcherAgent и ContrarianAgent) и добавь их в новую активную фазу.
4.  Если пробелов нет и все цели достигнуты, измени `main_goal_status` на `READY_FOR_FINAL_BRIEF`.
5.  Верни **полностью обновленный объект стратегического плана**.

Ты ОБЯЗАН вернуть результат в формате JSON.
"""
        updated_plan = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=StrategicPlan,
            budget_manager=self.budget_manager
        )
        if updated_plan and "phases" in updated_plan:
            print("   [Supervisor] Рефлексия завершена. План обновлен.")
            return updated_plan
        else:
            print("!!! Supervisor: Не удалось сгенерировать обновленный план.")
            return current_plan