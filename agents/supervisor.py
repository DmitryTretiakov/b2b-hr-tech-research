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
        Создает первоначальный план, верифицируя гипотезы и исследуя главную цель.
        """
        model_name = "gemini-2.5-pro"
        
        # Форматируем конфиг для чистой вставки в промпт
        config_str = yaml.dump(user_config, allow_unicode=True, sort_keys=False)

        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Ведущий Стратегический Аналитик и AI Product Owner. Твоя задача - создать исчерпывающий, но ресурсоэффективный план исследования на основе предоставленного брифа.

**ПОЛНЫЙ КОНТЕКСТ ПРОЕКТА (BRIEF):**
```yaml
{config_str}
```

**ТВОЯ ДВУЕДИНАЯ ЗАДАЧА:**

**1. ВЕРИФИКАЦИЯ ГИПОТЕЗ (Принцип "Не Доверяй, а Проверяй"):**
   - Проанализируй раздел `initial_hypotheses`. Каждое утверждение в нем - это гипотеза, а не факт.
   - Для **каждой ключевой гипотезы** создай состязательную пару задач:
     - Одна для `ResearcherAgent` (найти подтверждения).
     - Одна для `ContrarianAgent` (найти опровержения, альтернативные мнения, риски).
   - **Пример:** Для гипотезы "UI/UX LMS IDO устарел", создай задачи "Найти обзоры и отзывы, подтверждающие устарелость интерфейса Moodle/IDO" и "Найти примеры успешного использования Moodle в корпоративном секторе, опровергающие тезис об устарелости".

**2. СТРАТЕГИЧЕСКОЕ ИССЛЕДОВАНИЕ (Движение к Цели):**
   - Проанализируй `user_context.main_goal` и `project_context.product_vision`. Особое внимание удели **экономическому обоснованию, созданию дорожной карты и финансовой модели**.
   - Создай набор исследовательских задач для достижения этой цели. Включи задачи на поиск данных для **расчета CAPEX/OPEX, анализа бизнес-моделей конкурентов и составления Product Roadmap**.
   - Используй `additional_tasks` как прямое руководство к действию.

**ПРАВИЛА ФОРМИРОВАНИЯ ПЛАНА:**
- **Декомпозиция:** Разбивай сложные цели на простые, атомарные задачи.
- **ID Задач:** Используй префиксы `verify_` для задач верификации и `research_` для исследовательских задач. Например: `verify_01_researcher`, `research_01_contrarian`.
- **Ресурсоэффективность:** Для **ВСЕХ** задач в `initial_model_assignments` назначь самую дешевую подходящую модель: `'gemma-3'`. Эскалация будет происходить автоматически.

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
