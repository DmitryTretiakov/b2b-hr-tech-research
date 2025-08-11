# agents/meta_agents.py
import json
from agents.base_agent import BaseAgent
from core.tool_registry import ToolRegistry
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import ArchitectDecision, FinalAuditReport, ValidationReport, FailureAnalysisReport

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
        
        base_prompt = f"""
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


        max_retries = 2
        for attempt in range(max_retries):
            print(f"      [ToolSmith] Попытка генерации кода {attempt + 1}/{max_retries}...")
            
            current_prompt = base_prompt
            if attempt > 0:
                current_prompt += f"\n\n**ВАЖНО:** Твой предыдущий код не прошел проверку синтаксиса. Ошибка: `{syntax_error}`. Пожалуйста, исправь код и верни только валидный Python-код."

            response = self.llm_client.invoke("gemini-2.5-pro", current_prompt)
            code = response.content if hasattr(response, 'content') else ""
            
            # Очистка кода от Markdown
            if code.startswith("```python"):
                code = code[len("```python"):].strip()
            if code.endswith("```"):
                code = code[:-3].strip()

            # Проверка синтаксиса
            try:
                compile(code, f"{tool_name}.py", "exec")
                print("      [ToolSmith] <- Синтаксис кода успешно проверен.")
                print("   [ToolSmithAgent] <- Генерация инструмента завершена.")
                return code
            except SyntaxError as e:
                print(f"      [ToolSmith] !!! Ошибка синтаксиса в сгенерированном коде: {e}")
                syntax_error = e # Сохраняем ошибку для следующей итерации

        # Если все попытки провалились
        raise Exception(f"Не удалось сгенерировать синтаксически корректный код для инструмента '{tool_name}' после {max_retries} попыток.")
    

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

    def fix_or_enhance(self, failed_task: dict, validation_report: dict | None, failure_feedback: str | None) -> dict:
        """
        Пытается исправить проваленную задачу.
        Приоритет: фидбек от FailureAnalyst, затем отчет от Validator.
        """
        print(f"   [ArchitectAgent] -> Анализирую задачу '{failed_task.get('task_id')}'...")

        tool_description = None
        # Приоритет у фидбека от аналитика сбоев
        if failure_feedback:
            print("   [ArchitectAgent] Использую фидбек от FailureAnalyst для перегенерации инструмента.")
            tool_description = failure_feedback # Предполагаем, что фидбек - это и есть описание для ToolSmith
        elif validation_report and validation_report.get('missing_tool_description'):
            tool_description = validation_report.get('missing_tool_description')
        
        if not tool_description:
            print("   [ArchitectAgent] !!! Не удалось определить необходимый инструмент (отчет валидатора и фидбек отсутствуют).")
            print("   [ArchitectAgent] <- Причина сбоя, вероятно, не в инструментах. Не могу исправить автоматически.")
            failed_task['status'] = 'FATAL_ERROR'
            return failed_task
        
        # Генерируем имя для инструмента из его описания
        prompt_for_name = "Придумай короткое, но осмысленное имя в snake_case для инструмента, который делает следующее: '{}'. Верни только имя, например: 'search_and_read_webpage'.".format(tool_description)
        response = self.llm_client.invoke("gemini-2.5-flash", prompt_for_name)
        tool_name = response.content.strip().replace("`", "")

        print(f"   [ArchitectAgent] <- РЕШЕНИЕ: Создать новый инструмент '{tool_name}'.")
        try:
            tool_code = self.toolsmith.generate_tool_code(tool_name, tool_description)
            self.tool_registry.register_tool(tool_name, tool_code)
            
            failed_task['status'] = 'PENDING'
            return failed_task
        except Exception as e:
            print(f"   [ArchitectAgent] !!! Процесс создания инструмента провалился: {e}. Сигнализирую о фатальной ошибке.")
            failed_task['status'] = 'FATAL_ERROR'
            return failed_task
        
    def conduct_final_audit(self, state: dict) -> dict:
        """
        Проводит финальную проверку состояния системы на соответствие главной цели.
        """
        print(f"   [ArchitectAgent] -> Провожу финальный аудит системы...")
        model_name = "gemini-2.5-pro" # Для этой критической задачи нужна лучшая модель
        sanitizer_model = "gemini-2.5-flash"

        # Собираем ключевую информацию для принятия решения
        audit_context = {
            "main_goal": state.get("user_config", {}).get("user_context", {}).get("main_goal"),
            "artifacts_created": list(state.get("artifacts", {}).keys()),
            "knowledge_base_summary": {
                "total_facts": len(state.get("knowledge_base", {})),
                "key_topics": list(set(fact['claim_id'].split('_')[1] for fact in state.get("knowledge_base", {}).values()))
            }
        }
        context_str = json.dumps(audit_context, indent=2, ensure_ascii=False)

        prompt = f"""
**ТВОЯ РОЛЬ:** Главный Аудитор Проекта. Твоя задача - вынести финальный вердикт: достигнута ли главная цель проекта.

**КОНТЕКСТ ДЛЯ АУДИТА:**
```json
{context_str}
```

**ТВОЯ ЗАДАЧА:**
1.  **Сравни Цель и Результат:** Внимательно прочитай `main_goal`. Сравни ее с тем, что было реально сделано (`artifacts_created`, `knowledge_base_summary`).
2.  **Прими Решение:**
    *   Если ты считаешь, что созданные артефакты и собранные данные полностью отвечают на `main_goal`, установи `is_complete: true`.
    *   Если чего-то не хватает (например, цель была "создать фин. модель и презентацию", а создан только один артефакт), установи `is_complete: false`.
3.  **Обоснуй и Действуй:**
    *   В поле `reasoning` четко объясни свое решение.
    *   Если `is_complete: false`, в поле `new_tasks` предложи список **конкретных** задач, которые нужно выполнить, чтобы закрыть пробелы. Например: `[{{"task_id": "final_artifact_01", "agent_name": "RoadmapVisualizationAgent", "description": "Создать финальную диаграмму дорожной карты в формате Mermaid на основе существующих артефактов."}}]`.

Верни ТОЛЬКО JSON-объект, соответствующий схеме `FinalAuditReport`.
"""
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, sanitizer_model, prompt,
            FinalAuditReport, self.budget_manager
        )
        print(f"   [ArchitectAgent] <- Вердикт аудита: is_complete={report.get('is_complete')}")
        return report

        

class FailureAnalystAgent(BaseAgent):
    """
    Агент-диагност, который анализирует сбои в задачах и предлагает
    стратегию их исправления. Использует быструю и дешевую модель.
    """
    def execute(self, failed_task: dict, error_message: str, state: dict) -> dict:
        """
        Анализирует контекст сбоя и возвращает отчет с планом действий.
        """
        print(f"   [FailureAnalystAgent] -> Анализирую сбой в задаче '{failed_task.get('task_id')}'...")
        model_name = "gemini-2.5-flash"
        sanitizer_model = "gemini-2.5-flash-lite"

        context = {
            "failed_task": failed_task,
            "error_message": error_message,
            "model_used": state['model_assignments'].get(failed_task.get('task_id')),
            "escalation_count": state.get('escalation_count', 0),
            "available_tools": self.tool_registry.get_tools_for_prompt() if self.tool_registry else "No tools available."
        }
        context_str = json.dumps(context, indent=2, ensure_ascii=False)

        prompt = f"""
**ТВОЯ РОЛЬ:** Ведущий Инженер по Надежности Систем (SRE). Твоя задача - диагностировать сбой и предложить наилучший, наиболее экономичный способ его устранения.

**КОНТЕКСТ СБОЯ:**
```json
{context_str}
```

**ТВОЯ ЗАДАЧА - ПРОВЕСТИ АНАЛИЗ И ВЫБРАТЬ ОДНО ДЕЙСТВИЕ:**

1.  **Анализ Ошибки:** Внимательно изучи `error_message`.
    *   Если ошибка похожа на временный сетевой сбой, проблему с API или таймаут (`ConnectionError`, `Timeout`, `50x HTTP error`), выбери действие `RETRY`.
    *   Если ошибка явно указывает на исчерпание квот или лимитов, выбери `FATAL_ERROR`, так как дальнейшие попытки бессмысленны.
    *   Если ошибка связана с тем, что инструмент не смог найти данные (например, "не удалось найти информацию о зарплате"), это проблема в подходе. Выбери `CREATE_NEW_TASK` и переформулируй исходную задачу, сделав ее более общей или предложив другой подход (например, "Искать зарплату для Python Developer в России, а не только в Томске").
    *   Если ошибка связана с кодом самого инструмента (например, `TypeError`, `AttributeError` внутри инструмента), выбери `REGENERATE_TOOL` и в поле `feedback` дай четкие инструкции для `ArchitectAgent`, что именно нужно исправить в коде.
    *   Если предыдущие попытки уже провалились (`escalation_count` > 0) и ошибка не очевидна, возможно, стоит попробовать более мощную модель. Выбери `RETRY_WITH_NEW_MODEL`.
    *   Если ничего из вышеперечисленного не подходит или исправить ситуацию невозможно, выбери `FATAL_ERROR`.

2.  **Заполнение `data`:**
    *   Для `RETRY_WITH_NEW_MODEL`: Укажи следующую по мощности модель в `data.next_model_name`.
    *   Для `REGENERATE_TOOL`: Укажи фидбек в `data.feedback`.
    *   Для `CREATE_NEW_TASK`: Укажи новый, улучшенный текст задачи в `data.new_task_description`.

Верни ТОЛЬКО JSON-объект, соответствующий схеме `FailureAnalysisReport`.
"""

        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, sanitizer_model, prompt,
            FailureAnalysisReport, self.budget_manager
        )
        print(f"   [FailureAnalystAgent] <- Вердикт: {report.get('action')}. Причина: {report.get('reasoning')}")
        return report

