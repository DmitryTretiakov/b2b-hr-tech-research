# agents/supervisor.py
from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import GraphPlan
from typing import Dict
import yaml

if 'ToolRegistry' not in locals():
    from core.tool_registry import ToolRegistry
if 'LLMClient' not in locals():
    from core.llm_client import LLMClient
if 'APIBudgetManager' not in locals():
    from core.budget_manager import APIBudgetManager


class SupervisorAgent(BaseAgent):
    """
    Генерирует первоначальный и последующие планы для графа на основе богатого контекста.
    Использует модель Уровня 4 (Pro).
    """
    def __init__(self, llm_client: LLMClient, budget_manager: APIBudgetManager, tool_registry: ToolRegistry = None):
        super().__init__(llm_client, budget_manager, tool_registry)

    def create_initial_plan(self, user_config: Dict) -> Dict:
        """
        Создает первоначальный план, включая задачи на верификацию, исследование и создание артефактов.
        """
        model_name = "gemini-2.5-pro"
        
        # === ИЗМЕНЕНИЕ НАЧАТО: Упрощение промпта для повышения надежности ===
        # Вместо передачи всего конфига, передаем только ключевые секции.
        context_summary = {
            "main_goal": user_config.get("user_context", {}).get("main_goal"),
            "product_vision": user_config.get("project_context", {}).get("product_vision"),
            "initial_hypotheses": user_config.get("initial_hypotheses", {}).get("tsu_assets_analysis"),
            "additional_tasks": user_config.get("initial_hypotheses", {}).get("additional_tasks")
        }
        context_str = yaml.dump(context_summary, allow_unicode=True, sort_keys=False, indent=2)
        
        allowed_research_models = "'gemma-3', 'gemini-2.5-flash-lite', 'gemini-2.5-flash'"
        
        prompt = f"""
**ТВОЯ РОЛЬ:** Ведущий Стратегический Аналитик.

**КЛЮЧЕВЫЕ ДАННЫЕ ПРОЕКТА:**
```yaml
{context_str}```

**ТВОЯ ЗАДАЧА:** Создать детальный план действий в формате JSON.

**ИНСТРУКЦИИ ПО ВЫПОЛНЕНИЮ (два шага):**

**ШАГ 1: Продумай план (Твои мысли).**
Проанализируй ключевые данные и в свободной форме, как для себя, набросай список всех необходимых задач. Раздели их на две категории:
1.  **Задачи по сбору данных:** Верификация гипотез и выполнение дополнительных задач. Для них будет использоваться `SingleStepToolAgent`.
2.  **Задачи по созданию артефактов:** Создание финансовой модели (`FinancialModelAgent`) или дорожной карты (`ProductManagerAgent`).

**ШАГ 2: Отформатируй план в JSON.**
После того как ты продумал план, отформатируй его в виде ОДНОГО JSON-объекта, который строго соответствует схеме `GraphPlan`.
- Для каждой задачи из Шага 1 создай объект в списке `tasks` с `agent_name: "SingleStepToolAgent"`.
- Для каждой задачи по созданию артефактов создай соответствующий объект (`agent_name: "FinancialModelAgent"` и т.д.).
- Присвой каждой задаче уникальный `task_id` с префиксами `verify_`, `research_` или `artifact_`.
- Заполни словарь `initial_model_assignments`: для `SingleStepToolAgent` используй 'gemma-3', для остальных - 'gemini-2.5-flash'.

**Пример твоих мыслей (Шаг 1):**
"Окей, мне нужно проверить гипотезу о синергии - это будет задача для SingleStepToolAgent. Затем нужно найти зарплаты - еще одна задача для него. И, наконец, создать фин. модель - это для FinancialModelAgent."

**Финальный результат (Шаг 2) должен быть ТОЛЬКО JSON-объектом.**
"""
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

        print("   [SupervisorAgent] -> Генерирую контекстно-осознанный план графа...")
        plan_data = invoke_llm_for_json_with_retry(
            llm_client=self.llm_client,
            model_name=model_name,
            sanitizer_model_name="gemini-2.5-flash",
            prompt=prompt,
            pydantic_schema=GraphPlan,
            budget_manager=self.budget_manager
        )
        
        # === ИЗМЕНЕНИЕ НАЧАТО: Добавлен финальный предохранитель ===
        if not plan_data or not plan_data.get('tasks'):
            error_msg = "КРИТИЧЕСКАЯ ОШИБКА: SupervisorAgent не смог сгенерировать валидный план задач после всех попыток. Выполнение невозможно."
            print(f"   [SupervisorAgent] !!! {error_msg}")
            raise ValueError(error_msg)
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===
            
        print("   [SupervisorAgent] <- План графа успешно сгенерирован.")
        return plan_data


    def create_next_phase_plan(self, analysis_summary: dict) -> Dict:
        """Генерирует план для следующей фазы на основе анализа предыдущей."""
        model_name = "gemini-2.5-pro"
        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Главный Архитектор AI-систем. Предыдущая фаза исследования завершена. Твоя задача - спланировать СЛЕДУЮЩУЮ фазу.

**СВОДКА ПРЕДЫДУЩЕЙ ФАЗЫ (от AnalystAgent):**
- **Ключевые Выводы:** {analysis_summary.get('key_insights', 'Нет данных')}
- **Обнаруженные Пробелы в Данных:** {analysis_summary.get('data_gaps', 'Нет данных')}

**ТВОЯ ЗАДАЧА:**
1.  Проанализируй выводы и, что более важно, **пробелы в данных**.
2.  Сгенерируй новый набор исследовательских задач, нацеленных на **закрытие этих пробелов**.
3.  Если пробелов нет или они не критичны, верни пустой список задач `[]`.
4.  Для КАЖДОЙ задачи назначь **самую дешевую подходящую модель** (например, 'gemma-3').
5.  Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
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
