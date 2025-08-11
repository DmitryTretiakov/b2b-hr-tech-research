# agents/supervisor.py
from __future__ import annotations
import json
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import GraphPlan, Task
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
        Создает первоначальный план ТОЛЬКО для фазы сбора и анализа данных.
        """
        model_name = "gemini-2.5-pro"
        
        context_summary = {
            "main_goal": user_config.get("user_context", {}).get("main_goal"),
            "product_vision": user_config.get("project_context", {}).get("product_vision"),
            "initial_hypotheses": user_config.get("initial_hypotheses", {})
        }
        context_str = yaml.dump(context_summary, allow_unicode=True, sort_keys=False, indent=2)
        
        agent_contract = """
**УТВЕРЖДЕННЫЙ СПИСОК АГЕНТОВ ДЛЯ ФАЗЫ ИССЛЕДОВАНИЯ:**
- **`SingleStepToolAgent`**: Для выполнения простых задач по сбору информации.
- **`CompetitorAnalysisAgent`**: Для проведения структурированного анализа конкурентов.
- **`TechnologyDeepDiveAgent`**: Для проведения глубокого анализа технических аспектов.
"""

        prompt = f"""
**ТВОЯ РОЛЬ:** Ведущий Стратегический Архитектор.
**ТВОЯ ЗАДАЧА:** Проанализируй `main_goal` и `initial_hypotheses` и создай детальный план действий **ТОЛЬКО для первой фазы исследования: Сбор и Анализ Данных.**
Не включай в план задачи по созданию артефактов или написанию отчета. Эти задачи будут спланированы позже.

{agent_contract}

**КЛЮЧЕВЫЕ ДАННЫЕ ПРОЕКТА:**```yaml
{context_str}```

**ПРАВИЛА ПОСТРОЕНИЯ ПЛАНА:**
1.  **Зависимости по выполнению (`dependencies`):** Указывай ID задач, которые должны быть *завершены* перед началом текущей. Это управляет порядком выполнения.
2.  **Зависимости по данным (`data_dependencies`):** Указывай ID задач, чьи *результаты* необходимы для выполнения текущей задачи. Это управляет потоком данных.
    - Пример: Задача анализа конкурентов (`CompetitorAnalysisAgent`) должна иметь в `data_dependencies` ID задач, которые собирали информацию о конкурентах (`SingleStepToolAgent`).
3.  Задачи сбора данных (`SingleStepToolAgent`) обычно не имеют зависимостей.
4.  Задачи анализа (`CompetitorAnalysisAgent`, `TechnologyDeepDiveAgent`) всегда зависят от задач сбора данных.
5.  Для задач `SingleStepToolAgent` назначь модель 'gemma-3'.
6.  Для всех остальных агентов-специалистов назначь модель 'gemini-2.5-flash'.

**ФОРМАТ ВЫВОДА:**
Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
"""

        print("   [SupervisorAgent] -> Генерирую план для Фазы 1 (Исследование)...")
        plan_data = invoke_llm_for_json_with_retry(
            llm_client=self.llm_client,
            model_name=model_name,
            sanitizer_model_name="gemini-2.5-flash",
            prompt=prompt,
            pydantic_schema=GraphPlan,
            budget_manager=self.budget_manager
        )
        
        if not plan_data or not plan_data.get('tasks'):
            error_msg = "КРИТИЧЕСКАЯ ОШИБКА: SupervisorAgent не смог сгенерировать валидный план задач."
            print(f"   [SupervisorAgent] !!! {error_msg}")
            raise ValueError(error_msg)
            
        print("   [SupervisorAgent] <- План Фазы 1 успешно сгенерирован.")
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
    
    def create_artifact_and_report_plan(self, state: dict) -> Dict:
        """
        На основе наполненной Базы Знаний создает план для генерации артефактов и отчета.
        """
        model_name = "gemini-2.5-pro"
        
        context = {
            "main_goal": state.get("user_config", {}).get("user_context", {}).get("main_goal"),
            "knowledge_base_size": len(state.get("knowledge_base", {})),
        }
        context_str = json.dumps(context, indent=2, ensure_ascii=False)

        agent_contract = """
**УТВЕРЖДЕННЫЙ СПИСОК АГЕНТОВ ДЛЯ ФАЗЫ ГЕНЕРАЦИИ:**
- **`FinancialModelAgent`**: Создает артефакт с финансовой моделью.
- **`ProductManagerAgent`**: Создает артефакт с User Stories.
- **`RoadmapVisualizationAgent`**: Визуализирует дорожную карту.
- **`OutlineAgent`**: Создает план финального отчета.
- **`SectionWriterAgent`**: Пишет одну секцию отчета.
- **`ReportWriterAgent`**: Компилирует финальный отчет.
"""
        prompt = f"""
**ТВОЯ РОЛЬ:** Ведущий Стратегический Архитектор.
**КОНТЕКСТ:** Фаза исследования завершена, База Знаний наполнена.
**ТВОЯ ЗАДАЧА:** Спланировать **финальную фазу: Создание Артефактов и Написание Отчета.**

{agent_contract}

**КЛЮЧЕВЫЕ ДАННЫЕ ПРОЕКТА:**
```json
{context_str}
```

**ПРАВИЛА:**
- Проанализируй `main_goal` и создай задачи для генерации всех необходимых артефактов.
- Создай полную цепочку задач для написания отчета: одна задача для `OutlineAgent`, затем несколько для `SectionWriterAgent` (по одной на каждую предполагаемую секцию), и одна финальная для `ReportWriterAgent`.
- Установи правильные зависимости (`dependencies`). Например, `RoadmapVisualizationAgent` должен зависеть от `ProductManagerAgent`. Задачи `SectionWriterAgent` должны зависеть от `OutlineAgent`.
- Для всех агентов назначь модель 'gemini-2.5-flash', кроме `ReportWriterAgent` (для него 'gemini-2.5-pro').

**ФОРМАТ ВЫВОДА:**
Верни результат в виде ОДНОГО JSON-объекта, соответствующего схеме `GraphPlan`.
"""
        print("   [SupervisorAgent] -> Генерирую план для Фазы 2 (Артефакты и Отчет)...")
        plan_data = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt, GraphPlan, self.budget_manager
        )
        if not plan_data:
            return {"tasks": [], "initial_model_assignments": {}}
            
        print(f"   [SupervisorAgent] <- План Фазы 2 сгенерирован ({len(plan_data.get('tasks', []))} задач).")
        return plan_data

