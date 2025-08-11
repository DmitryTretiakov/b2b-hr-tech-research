# agents/meta_agents.py
import json
from agents.base_agent import BaseAgent
from core.tool_registry import ToolRegistry
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import ArchitectDecision, ValidationReport

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
    Агент, ответственный за динамическое создание новых инструментов.
    """
    def generate_tool_code(self, tool_name: str, tool_description: str) -> str:
        """
        Генерирует Python-код для нового инструмента на основе его описания.
        """
        print(f"   [ToolSmithAgent] -> Генерирую код для инструмента: {tool_description[:100]}...")
        
        prompt = f"""
Твоя роль: Старший Python-разработчик, специализирующийся на написании надежных, изолированных инструментов, работающих в строго контролируемом окружении.
Твоя задача: Написать код для нового инструмента.

**ИМЯ ИНСТРУМЕНТА:** `{tool_name}`
**ОПИСАНИЕ ЗАДАЧИ ИНСТРУМЕНТА:** {tool_description}

**КРИТИЧЕСКИ ВАЖНЫЕ ТРЕБОВАНИЯ К КОДУ:**

1.  **ОГРАНИЧЕННЫЕ ЗАВИСИМОСТИ:** Ты можешь использовать **ТОЛЬКО** следующие библиотеки. Попытка импортировать что-либо другое приведет к сбою.
    - `import requests`
    - `from bs4 import BeautifulSoup`
    - Стандартные библиотеки Python (`os`, `json`, `re`, `time`, etc.)
    **ЗАПРЕЩЕНО:** `googlesearch`, `duckduckgo_search` и любые другие сторонние поисковые библиотеки.

2.  **СТРУКТУРА КОДА:**
    - Код должен быть в одном файле и содержать ТОЛЬКО ОДНУ функцию.
    - **ИМЯ ФУНКЦИИ ДОЛЖНО БЫТЬ В ТОЧНОСТИ `{tool_name}`.** Это не обсуждается.
    - Функция должна иметь type hints для всех аргументов и возвращаемого значения.
    - Функция должна иметь подробный docstring.

3.  **ОБРАБОТКА ОШИБОК:**
    - Если инструмент не может выполнить свою задачу, он должен вызывать исключение (`raise Exception(...)`).
    - **ЗАПРЕЩЕНО:** Использовать пустые `except:`. Всегда используй `except Exception as e`, чтобы ошибка была корректно обработана.
**КРИТИЧЕСКИЕ ОГРАНИЧЕНИЯ И АНТИ-ПАТТЕРНЫ (ИЗБЕГАЙ ИХ):**

1.  **АНТИ-ПАТТЕРН: Прямой скрейпинг страниц результатов поисковых систем (Google, Yandex и т.д.).**
    - **ПОЧЕМУ ЭТО ПЛОХО:** Их HTML-структура постоянно меняется, они активно блокируют ботов. Это крайне ненадежно.
    - **ПРАВИЛЬНЫЙ ПОДХОД:** Если тебе нужно что-то найти, используй поисковый API (например, Google Custom Search или Serper), чтобы получить список URL-адресов. Затем читай контент с этих URL-адресов.

2.  **АНТИ-ПАТТЕРН: Чрезмерная зависимость от регулярных выражений для извлечения сложных данных.**
    - **ПОЧЕМУ ЭТО ПЛОХО:** Регулярные выражения хороши для простых форматов, но они не понимают контекст. Например, извлекая зарплату, они могут ошибочно вытащить номер телефона или другую цифру.
    - **ПРАВИЛЬНЫЙ ПОДХОД:** Для извлечения данных, требующих понимания смысла (например, "найди зарплату", "найди имя CEO"), используй многошаговую логику: получи текст со страницы, а затем сделай внутренний, узконаправленный вызов к LLM с просьбой извлечь нужную информацию из этого текста в формате JSON.
    
**ПРИМЕР ПРАВИЛЬНОЙ СТРУКТУРЫ:**
```python
# Разрешенные импорты
import requests
from bs4 import BeautifulSoup
import json

def {tool_name}(query: str) -> dict:
    \"\"\"
    Подробный docstring, объясняющий все.
    \"\"\"
    try:
        # Логика функции с использованием ТОЛЬКО разрешенных библиотек
        # ...
        return {{"status": "success", "data": "some_value"}}
    except requests.exceptions.RequestException as e:
        raise Exception(f"Сетевая ошибка: {{e}}")
    except Exception as e:
        raise Exception(f"Неизвестная ошибка: {{e}}")
```

Напиши полный код для инструмента {tool_name}. Верни только сам код, обернутый в блок python ....
"""


        
        response = self.llm_client.invoke("gemini-2.5-pro", prompt)
        code = response.content if hasattr(response, 'content') else ""
        
        if code.startswith("```python"):
            code = code[len("```python"):].strip()
        if code.endswith("```"):
            code = code[:-3].strip()
            
        print("   [ToolSmithAgent] <- Генерация инструмента завершена.")
        return code
    

class ValidatorAgent(BaseAgent):
    """
    Проактивно проверяет, можно ли выполнить задачу с текущим набором инструментов,
    прежде чем передавать ее на исполнение.
    """
    def execute(self, task: dict) -> dict:
        """
        Выполняет валидацию задачи.
        """
        print(f"   [ValidatorAgent] -> Проверяю задачу '{task.get('task_id')}' на исполнимость...")
        
        if not self.tool_registry:
             raise ValueError("ValidatorAgent требует доступа к ToolRegistry.")

        available_tools = self.tool_registry.get_tools_for_prompt()
        prompt = f"""
Твоя роль: Скрупулезный системный аналитик-планировщик. Твоя задача - предотвратить бессмысленную работу и упростить ее для исполнителей.

**ЗАДАЧА ДЛЯ АНАЛИЗА:**
{task.get('description')}

**ДОСТУПНЫЕ ИНСТРУМЕНТЫ:**
{available_tools}

**ИНСТРУКЦИИ:**
1.  Внимательно прочитай описание задачи и сравни с возможностями инструментов.
2.  **Сценарий 1: Задача НЕВЫПОЛНИМА.** Если для выполнения задачи очевидно не хватает инструмента (например, нужно прочитать файл, а инструмента нет), установи `is_executable: false` и **обязательно** заполни `missing_tool_description`.
3.  **Сценарий 2: Задача ВЫПОЛНИМА.** Если задача выполнима с помощью имеющихся инструментов, установи `is_executable: true`.
4.  **Критически важно для Сценария 2:** Если задача требует нескольких шагов (например, сначала поиск, а потом чтение каждой найденной страницы), ты **обязан** предоставить простой пошаговый план в поле `suggested_plan`. Например: ["Сначала используй web_search с запросом 'X'", "Затем для каждой релевантной ссылки вызови webpage_reader", "Проанализируй полученные тексты"]. Если задача простая и требует одного шага, оставь `suggested_plan` пустым.

Верни ТОЛЬКО JSON-объект, соответствующий схеме `ValidationReport`.
"""
        
        report = invoke_llm_for_json_with_retry(
            self.llm_client,
            "gemini-2.5-flash", # Используем быструю модель, как и было запрошено
            "gemini-2.5-flash-lite",
            prompt,
            ValidationReport,
            self.budget_manager
        )
        print(f"   [ValidatorAgent] <- Вердикт: is_executable={report.get('is_executable')}")
        return report

class ArchitectAgent(BaseAgent):
    """
    Мета-агент, отвечающий за самокоррекцию графа.
    Теперь он реагирует на вердикт Валидатора.
    """
    def __init__(self, llm_client, budget_manager, tool_registry: ToolRegistry, toolsmith: ToolSmithAgent):
        super().__init__(llm_client, budget_manager, tool_registry)
        self.toolsmith = toolsmith

    def fix_or_enhance(self, failed_task: dict, validation_report: dict | None) -> dict:
        """
        Пытается исправить проваленную задачу. Основная стратегия - создание инструмента.
        Теперь устойчив к отсутствию validation_report.
        """
        print(f"   [ArchitectAgent] -> Анализирую задачу '{failed_task.get('task_id')}'...")

        # === ИЗМЕНЕНИЕ НАЧАТО: Добавлена проверка на наличие отчета и описания инструмента ===
        tool_description = None
        if validation_report and validation_report.get('missing_tool_description'):
            tool_description = validation_report.get('missing_tool_description')
        
        if not tool_description:
            print("   [ArchitectAgent] !!! Не удалось определить необходимый инструмент (отчет валидатора отсутствует или пуст).")
            print("   [ArchitectAgent] <- Причина сбоя, вероятно, не в инструментах (например, ошибка API или логики агента). Не могу исправить автоматически.")
            failed_task['status'] = 'FATAL_ERROR'
            return failed_task
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

        # Генерируем имя для инструмента из его описания
        prompt_for_name = "Придумай короткое, но осмысленное имя в snake_case для инструмента, который делает следующее: '{}'. Верни только имя, например: 'search_and_read_webpage'.".format(tool_description)
        response = self.llm_client.invoke("gemini-2.5-flash", prompt_for_name)
        tool_name = response.content.strip().replace("`", "")

        print(f"   [ArchitectAgent] <- РЕШЕНИЕ: Создать новый инструмент '{tool_name}'.")
        try:
            tool_code = self.toolsmith.generate_tool_code(tool_name, tool_description)
            self.tool_registry.register_tool(tool_name, tool_code)
            
            # Возвращаем задачу в очередь для повторного выполнения с новым инструментом
            failed_task['status'] = 'PENDING'
            return failed_task
        except Exception as e:
            print(f"   [ArchitectAgent] !!! Процесс создания инструмента провалился: {e}. Сигнализирую о фатальной ошибке.")
            failed_task['status'] = 'FATAL_ERROR'
            return failed_task
