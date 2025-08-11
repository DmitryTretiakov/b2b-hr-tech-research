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
        
        agent_contract = """
**УТВЕРЖДЕННЫЙ СПИСОК АГЕНТОВ-ИСПОЛНИТЕЛЕЙ (ТВОЙ ВЫБОР ОГРАНИЧЕН ТОЛЬКО ИМИ):**

**Часть 1: Сбор и Анализ Данных**
- **`SingleStepToolAgent`**: Выполняет простые задачи по сбору информации с помощью инструментов.
- **`CompetitorAnalysisAgent`**: Проводит структурированный анализ конкурентов.
- **`TechnologyDeepDiveAgent`**: Проводит глубокий анализ технических аспектов.

**Часть 2: Создание Промежуточных Артефактов**
- **`FinancialModelAgent`**: Создает артефакт с финансовой моделью.
- **`ProductManagerAgent`**: Создает артефакт с User Stories и эпиками.
- **`RoadmapVisualizationAgent`**: Визуализирует дорожную карту (использовать ПОСЛЕ `ProductManagerAgent`).

**Часть 3: Стратегическое Планирование Финальных Документов**
У тебя есть несколько мощных агентов для создания финальных отчетов. Твоя задача — не просто выбрать одного, а **спроектировать наилучшую стратегию** их использования, исходя из `main_goal`.

**ТВОЙ ИНСТРУМЕНТАРИЙ ДЛЯ ОТЧЕТОВ:**
- **Агенты для быстрых, сфокусированных записок:**
  - `ProductOwnerMemoAgent`: Создает записку с акцентом на **продукт**.
  - `InvestmentMemoAgent`: Создает записку с акцентом на **экономику и инвестиции**.
- **Конвейер для глубоких, детализированных отчетов:**
  - `OutlineAgent` -> `SectionWriterAgent` (несколько задач) -> `ReportWriterAgent`.

**ПРИМЕРЫ СТРАТЕГИЙ (используй их как вдохновение, а не как жесткое правило):**

- **Простая Стратегия (Быстрый Меморандум):** Если `main_goal` требует одного сфокусированного документа, создай **одну** задачу для `ProductOwnerMemoAgent` или `InvestmentMemoAgent`.

- **Комплексная Стратегия (Детальный Отчет):** Если `main_goal` требует максимальной убедительности и детализации, спроектируй **полную цепочку** задач: сначала для `OutlineAgent`, затем несколько задач для `SectionWriterAgent` (по одной на каждую секцию из плана), и в конце одну задачу для `ReportWriterAgent`.

- **Гибридная Стратегия (Отчет + Резюме):** Если требуется и детальный анализ, и краткая выжимка для руководства, ты можешь запланировать **сначала создание полного отчета (Комплексная Стратегия), а затем, как финальный шаг, поручить `InvestmentMemoAgent` написать `executive_summary` на основе уже готового отчета.**
"""

        prompt = f"""
**ТВОЯ РОЛЬ:** Ведущий Стратегический Архитектор. Твоя задача - создать полный и логичный план, гибко комбинируя утвержденных агентов для наилучшего результата.

{agent_contract}

**КЛЮЧЕВЫЕ ДАННЫЕ ПРОЕКТА:**
```yaml
{context_str}```

**ТВОЯ ЗАДАЧА:**
Проанализируй `main_goal`. Спроектируй и создай полный план действий от сбора данных до создания финальных документов. Выбери или скомбинируй стратегии из Части 3, чтобы наилучшим образом соответствовать цели проекта.

**ПРАВИЛА РЕСУРСОЭФФЕКТИВНОСТИ:**
- Для задач `SingleStepToolAgent` назначь модель 'gemma-3'.
- Для всех остальных агентов-специалистов назначь модель 'gemini-2.5-flash'.

**ПРАВИЛА ПОСТРОЕНИЯ ЗАВИСИМОСТЕЙ (dependencies):**
- Задача сбора данных (`data_collection_...`) не должна зависеть ни от чего.
- Задача анализа (`analysis_...`) должна зависеть от всех релевантных задач сбора данных.
- Задача создания артефакта (`artifact_...`) должна зависеть от всех релевантных задач анализа и сбора данных.
- Задачи должны образовывать логическую последовательность.


**ФОРМАТ ВЫВОДА:**
Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
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
