# agents/supervisor.py
from __future__ import annotations
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
        
        config_str = yaml.dump(user_config, allow_unicode=True, sort_keys=False)

        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Ведущий Стратегический Аналитик и AI Product Owner. Твоя задача - создать исчерпывающий, но ресурсоэффективный план исследования и генерации артефактов на основе предоставленного брифа.

**ПОЛНЫЙ КОНТЕКСТ ПРОЕКТА (BRIEF):**
```yaml
{config_str}
```

**ТВОЯ ЗАДАЧА СОСТОИТ ИЗ ТРЕХ ЧАСТЕЙ:**

**1. ВЕРИФИКАЦИЯ ГИПОТЕЗ (Принцип "Не Доверяй, а Проверяй"):**
   - Проанализируй раздел `initial_hypotheses`. Каждое утверждение в нем - это гипотеза.
   - Для **каждой ключевой гипотезы** создай состязательную пару задач: одна для `ResearcherAgent` (найти подтверждения) и одна для `ContrarianAgent` (найти опровержения).

**2. СТРАТЕГИЧЕСКОЕ ИССЛЕДОВАНИЕ (Движение к Цели):**
   - Проанализируй `user_context.main_goal` и `project_context.product_vision`.
   - Создай набор исследовательских задач для сбора данных, необходимых для **экономического обоснования, создания дорожной карты и финансовой модели**.
   - Используй `additional_tasks` как прямое руководство к действию.

**3. ПЛАНИРОВАНИЕ АРТЕФАКТОВ (Финальный Синтез):**
   - Проанализируй `project_context.product_vision` и `main_goal` на предмет требований к созданию конкретных бизнес-артефактов.
   - Если требуется **финансовая модель**, создай задачу для `FinancialModelAgent`. Пример описания: "Создать базовую финансовую модель и юнит-экономику для продукта 'Карьерный Навигатор'".
   - Если требуется **дорожная карта или User Stories**, создай задачу для `ProductManagerAgent`. Пример описания: "Сгенерировать User Stories для MVP продукта 'Карьерный Навигатор'".

**ПРАВИЛА ФОРМИРОВАНИЯ ПЛАНА:**
- **ID Задач:** Используй префиксы: `verify_` для верификации, `research_` для исследования и `artifact_` для генерации артефактов.
- **Агенты:** Четко указывай `agent_name` для каждой задачи (`ResearcherAgent`, `ContrarianAgent`, `FinancialModelAgent`, `ProductManagerAgent`).
- **Ресурсоэффективность:** Для **ВСЕХ** задач в `initial_model_assignments` назначь самую дешевую подходящую модель: `'gemma-3'` для исследовательских задач и `'gemini-2.5-flash'` для задач по созданию артефактов, так как они требуют большего синтеза.

**ФОРМАТ ВЫВОДА:**
Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
"""
        print("   [SupervisorAgent] -> Генерирую контекстно-осознанный план графа...")
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
