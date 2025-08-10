# agents/meta_agents.py
from __future__ import annotations
import json
from pydantic import BaseModel, Field
from typing import Dict, List

# Используем forward reference hints для циклических зависимостей типов
if 'LLMClient' not in locals():
    from core.llm_client import LLMClient
if 'APIBudgetManager' not in locals():
    from core.budget_manager import APIBudgetManager
if 'ToolRegistry' not in locals():
    from core.tool_registry import ToolRegistry

from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import JanitorReport, ArchitectDecision


class ArchitectAgent(BaseAgent):
    """
    Мета-агент для самокоррекции системы. Диагностирует причину провала задачи
    и либо исправляет ее описание, либо инициирует создание нового инструмента.
    Использует модель Уровня 4 (Pro).
    """
    def __init__(self, llm_client: LLMClient, budget_manager: APIBudgetManager, tool_smith: 'ToolSmithAgent', tool_registry: ToolRegistry):
        # Обновленный вызов super() для передачи tool_registry
        super().__init__(llm_client, budget_manager, tool_registry)
        self.tool_smith = tool_smith
        print(f"-> Агент '{self.__class__.__name__}' инициализирован с доступом к ToolSmith.")

    def fix_or_enhance(self, failed_task: dict, error_history: str) -> dict:
        """
        Анализирует провал и решает: исправить промпт или создать инструмент.
        Возвращает задачу, которую нужно повторно поставить в очередь.
        """
        model_name = "gemini-2.5-pro"
        print(f"   [ArchitectAgent] -> Диагностирую провал задачи {failed_task.get('task_id')}...")
        
        available_tools = self.tool_registry.get_tools_for_prompt()

        prompt = f"""
**ТВОЯ РОЛЬ:** Главный Архитектор и Системный Интегратор AI-систем.
**ПРОБЛЕМА:** Задача ниже провалилась, даже после эскалации моделей.

**ПРОВАЛЕННАЯ ЗАДАЧА (JSON):**
```json
{json.dumps(failed_task, ensure_ascii=False, indent=2)}
```

**ИСТОРИЯ ОШИБОК:**
{error_history}

**ДОСТУПНЫЕ ИНСТРУМЕНТЫ:**
{available_tools}

**ТВОЯ ГЛАВНАЯ ЗАДАЧА: ДИАГНОСТИРОВАТЬ И ИСПРАВИТЬ СИСТЕМУ.**
Проанализируй задачу, ошибки и список доступных инструментов. Определи КОРНЕВУЮ ПРИЧИНУ провала.
Затем выбери ОДНО из двух действий:

1.  **`FIX_DESCRIPTION`**: Если задача сформулирована нечетко, двусмысленно или слишком широко, и ее можно выполнить с помощью **уже существующих** инструментов.
2.  **`CREATE_TOOL`**: Если для выполнения задачи **очевидно не хватает** специфического инструмента (например, нужен доступ к API, которого нет, или возможность читать определенный формат файлов).

**ФОРМАТ ВЫВОДА:**
Верни JSON-объект, который ТОЧНО соответствует одной из двух схем:
- `{{ "action": "FIX_DESCRIPTION", "data": {{ "new_description": "Новое, предельно конкретное описание задачи." }} }}`
- `{{ "action": "CREATE_TOOL", "data": {{ "tool_name": "имя_инструмента_в_snake_case", "tool_description": "Четкое и однозначное описание того, что должен делать инструмент, для другого AI-разработчика." }} }}`
"""
        decision_data = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            ArchitectDecision, self.budget_manager
        )

        if not decision_data:
            print(f"   [ArchitectAgent] !!! Не удалось принять решение. Возвращаю оригинальную задачу для повторной попытки.")
            return failed_task

        action = decision_data.get("action")
        data = decision_data.get("data")

        if action == "FIX_DESCRIPTION":
            print(f"   [ArchitectAgent] <- РЕШЕНИЕ: Исправить описание задачи.")
            original_task = failed_task.copy()
            original_task["description"] = data.get("new_description", original_task["description"])
            return original_task
        
        elif action == "CREATE_TOOL":
            tool_name = data.get("tool_name")
            tool_description = data.get("tool_description")
            print(f"   [ArchitectAgent] <- РЕШЕНИЕ: Создать новый инструмент '{tool_name}'.")
            
            if not tool_name or not tool_description:
                print(f"   [ArchitectAgent] !!! Недостаточно данных для создания инструмента. Возвращаю задачу.")
                return failed_task

            try:
                # Шаг 1: Генерируем код инструмента
                tool_code = self.tool_smith.create_tool(tool_description)
                # Шаг 2: Регистрируем инструмент в системе
                self.tool_registry.register_tool(tool_name, tool_code)
                # Шаг 3: Возвращаем ОРИГИНАЛЬНУЮ задачу в очередь. Теперь ее можно будет выполнить с новым инструментом.
                print(f"   [ArchitectAgent] Новый инструмент '{tool_name}' создан. Задача будет выполнена повторно.")
                return failed_task
            except Exception as e:
                print(f"   [ArchitectAgent] !!! Процесс создания инструмента провалился: {e}. Возвращаю задачу.")
                return failed_task
        
        return failed_task

class KnowledgeJanitorAgent(BaseAgent):
    """
    Агент для поддержания чистоты и актуальности Базы Знаний.
    Использует модель Уровня 3 (Flash).
    """
    def cleanup_knowledge_base(self, knowledge_base: dict) -> dict:
        model_name = "gemini-2.5-flash"
        print(f"   [KnowledgeJanitorAgent] -> Провожу очистку Базы Знаний ({len(knowledge_base)} фактов)...")
        
        if len(knowledge_base) < 5: # Нет смысла запускать на маленькой базе
            print("   [KnowledgeJanitorAgent] <- База знаний слишком мала для очистки.")
            return knowledge_base

        prompt = f"""
**ТВОЯ РОЛЬ:** Ты "Архивариус" и "Детектив Противоречий". Твоя задача - поддерживать чистоту в Базе Знаний.

**ПРАВИЛА АНАЛИЗА:**
1.  **Поиск Противоречий:** Найди факты, которые прямо или косвенно противоречат друг другу (например, разные зарплаты для одной и той же должности в одном городе).
2.  **Поиск Устаревших Данных:** Найди факты, которые являются более старыми или менее конкретными версиями других фактов. Архивировать следует менее ценный факт.

**БАЗА ЗНАНИЙ ДЛЯ АНАЛИЗА:**
```json
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
```

**ТВОЯ ЗАДАЧА:**
Проанализируй базу и верни JSON-отчет со следующими полями:
- `conflicts_found`: Список, где каждый элемент - это список ID конфликтующих фактов.
- `archived_ids`: Список ID фактов, которые следует пометить как 'ARCHIVED'.
"""
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash-lite", prompt,
            JanitorReport, self.budget_manager
        )

        if not report:
            print("   [KnowledgeJanitorAgent] !!! Не удалось сгенерировать отчет об очистке.")
            return knowledge_base

        # Применяем изменения к Базе Знаний
        updated_kb = knowledge_base.copy()
        archived_count = 0
        for fact_id in report.get('archived_ids', []):
            if fact_id in updated_kb and updated_kb[fact_id].get('status') == 'ACTIVE':
                updated_kb[fact_id]['status'] = 'ARCHIVED'
                archived_count += 1
        
        print(f"   [KnowledgeJanitorAgent] <- Очистка завершена. Найдено конфликтов: {len(report.get('conflicts_found', []))}. Заархивировано фактов: {archived_count}.")
        return updated_kb

class ToolSmithAgent(BaseAgent):
    """
    Агент для генерации Python-кода для новых инструментов.
    Использует модель Уровня 4 (Pro).
    """
    def create_tool(self, tool_description: str) -> str:
        model_name = "gemini-2.5-pro"
        print(f"   [ToolSmithAgent] -> Генерирую код для инструмента: {tool_description}...")
        
        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - ведущий Python-разработчик, специализирующийся на создании надежных, самодостаточных инструментов.

**ЗАДАЧА:** Напиши Python-код для функции, которая выполняет следующее: "{tool_description}".

**СТРОГИЕ ТРЕБОВАНИЯ К КОДУ:**
1.  **Самодостаточность:** Код должен содержать все необходимые импорты.
2.  **Одна Функция:** Результатом должен быть код ОДНОЙ функции. Имя функции должно быть в snake_case и соответствовать будущему названию инструмента.
3.  **Типизация:** Используй строгую типизацию Python (type hints).
4.  **Документация:** Напиши подробный docstring, объясняющий, что делает функция, ее параметры и что она возвращает. Это критически важно для других агентов.
5.  **Обработка Ошибок:** Включи базовую обработку ошибок (`try...except`).
6.  **Безопасность:** НЕ ИСПОЛЬЗУЙ `eval()`, `exec()` или `os.system()`.

Верни ТОЛЬКО Python-код в виде одной строки или блока кода. Никаких объяснений до или после.
"""
        response = self.llm_client.invoke(model_name, prompt)
        
        # Очистка от markdown-блоков, если модель их добавила
        code = response.content.strip()
        if code.startswith("```python"):
            code = code[9:]
        if code.endswith("```"):
            code = code[:-3]
        
        print(f"   [ToolSmithAgent] <- Генерация инструмента завершена.")
        return code.strip()
