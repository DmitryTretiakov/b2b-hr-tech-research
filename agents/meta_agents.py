# agents/meta_agents.py
import json
from pydantic import BaseModel, Field
from typing import Dict, List
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import JanitorReport

# Pydantic модель для валидации вывода ArchitectAgent
class FixedTask(BaseModel):
    task_id: str = Field(description="Оригинальный ID задачи.")
    agent_name: str = Field(description="Имя агента-исполнителя.")
    description: str = Field(description="Новое, исправленное и более четкое описание задачи.")

class ArchitectAgent(BaseAgent):
    """
    Мета-агент для самокоррекции системы. Вызывается при полной неудаче эскалации.
    Использует модель Уровня 4 (Pro).
    """
    def fix_task(self, failed_task: dict, error_history: str) -> dict:
        model_name = "gemini-2.5-pro"
        print(f"   [ArchitectAgent] -> Анализирую провал задачи {failed_task.get('task_id')}...")
        
        prompt = f"""
**ТВОЯ РОЛЬ:** Главный Архитектор AI-систем. Ты "системный отладчик".
**ПРОБЛЕМА:** Задача ниже провалилась несколько раз, даже после эскалации моделей.

**ПРОВАЛЕННАЯ ЗАДАЧА (JSON):**
```json
{json.dumps(failed_task, ensure_ascii=False, indent=2)}
```

**ИСТОРИЯ ОШИБОК:**
{error_history}

**ТВОЯ ЗАДАЧА:**
1.  Проанализируй описание задачи и ошибки. Вероятная причина - неоднозначная или слишком широкая формулировка.
2.  Перепиши поле `description` задачи. Сделай его **более конкретным, узким и однозначным**.
3.  Не меняй `task_id` и `agent_name`.
4.  Верни ПОЛНЫЙ, ИСПРАВЛЕННЫЙ объект задачи в формате JSON.
"""
        fixed_task_data = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            FixedTask, self.budget_manager
        )

        if not fixed_task_data:
            print(f"   [ArchitectAgent] !!! Не удалось сгенерировать исправление. Возвращаю оригинальную задачу.")
            return failed_task
        
        # Обновляем оригинальную задачу новыми данными
        original_task = failed_task.copy()
        original_task.update(fixed_task_data)
        
        print(f"   [ArchitectAgent] <- План исправления для задачи {original_task['task_id']} сгенерирован.")
        return original_task

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
            if fact_id in updated_kb and updated_kb[fact_id]['status'] == 'ACTIVE':
                updated_kb[fact_id]['status'] = 'ARCHIVED'
                archived_count += 1
        
        print(f"   [KnowledgeJanitorAgent] <- Очистка завершена. Найдено конфликтов: {len(report.get('conflicts_found', []))}. Заархивировано фактов: {archived_count}.")
        return updated_kb

class ToolSmithAgent(BaseAgent):
    """
    Агент для генерации новых инструментов "на лету".
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
2.  **Одна Функция:** Результатом должен быть код ОДНОЙ функции.
3.  **Типизация:** Используй строгую типизацию Python (type hints).
4.  **Документация:** Напиши подробный docstring, объясняющий, что делает функция, ее параметры и что она возвращает.
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
