# agents/worker.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.search_agent import SearchAgent
from core.budget_manager import APIBudgetManager
from utils.helpers import format_search_results_for_llm, invoke_llm_for_json_with_retry
from agents.models import SearchQueries, ClaimList

class WorkerAgent:
    """
    Базовый класс для агентов, генерирующих сырые данные.
    Теперь работает с пакетами задач для повышения эффективности.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, search_agent: SearchAgent, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.search_agent = search_agent
        self.budget_manager = budget_manager
        self.role_prompt = ""

    def _decompose_tasks_batch(self, tasks: list[dict]) -> dict:
        """Генерирует поисковые запросы для ПАКЕТА задач одним вызовом LLM."""
        print(f"   [{self.__class__.__name__}] Шаг 1/3: Генерирую поисковые запросы для {len(tasks)} задач...")
        
        tasks_description = "\n".join([f"- Task ID: {t['task_id']}, Description: {t['description']}" for t in tasks])

        prompt = f"""**ТВОЯ РОЛЬ:** Ты - ассистент-исследователь.
{self.role_prompt}

**ПАКЕТ ЗАДАЧ ДЛЯ ДЕКОМПОЗИЦИИ:**
{tasks_description}

**ИНСТРУКЦИИ:**
Для КАЖДОЙ задачи из пакета, сгенерируй от 3 до 5 поисковых запросов.
Верни результат как ОДИН JSON-объект, где ключи - это `task_id`, а значения - списки строк-запросов.
Пример: {{ "task_001": ["запрос 1", "запрос 2"], "task_002": ["запрос 3", "запрос 4"] }}
"""
        # Для этой задачи не нужна Pydantic-модель, так как структура динамическая
        # Мы будем использовать более простой вызов LLM и парсить JSON вручную.
        # Это упрощение, но для данной структуры оно оправдано.
        # В реальном проекте можно создать Pydantic модель для этого.
        response = self.llm.invoke(prompt)
        self.budget_manager.record_spend(self.llm.model)
        try:
            # Простой парсинг JSON из ответа
            return json.loads(response.content)
        except json.JSONDecodeError:
            print(f"!!! {self.__class__.__name__}: Не удалось распарсить JSON с поисковыми запросами.")
            return {}

    def _create_draft_claims_batch(self, tasks_with_search_results: list[dict]) -> list[dict]:
        """Создает черновой список "Утверждений" для пакета задач одним вызовом LLM."""
        print(f"   [{self.__class__.__name__}] Шаг 2/3: Создаю черновик утверждений для {len(tasks_with_search_results)} задач...")
        
        # Форматируем входные данные для большого промпта
        formatted_input = []
        for item in tasks_with_search_results:
            formatted_input.append(f"""
---
**Task ID:** {item['task']['task_id']}
**Task Description:** {item['task']['description']}
**Search Results:**
{item['search_results_str']}
---
""")
        
        prompt = f"""**ТВОЯ РОЛЬ:** Ты - {self.__class__.__name__}.
{self.role_prompt}

**ТВОЯ ЗАДАЧА:** Проанализируй результаты поиска для КАЖДОЙ из предоставленных задач и сгенерируй список "Утверждений" (Claims).

**ПАКЕТ ЗАДАЧ И РЕЗУЛЬТАТОВ ПОИСКА:**
{''.join(formatted_input)}

**ИНСТРУКЦИИ:**
- Для каждой задачи сгенерируй НЕ БОЛЕЕ 5-7 самых важных утверждений.
- Статус каждого утверждения должен быть 'UNVERIFIED'.
- Верни результат как ОДИН JSON-объект со списком `claims`.
- **ВАЖНО:** В каждом объекте Claim должно быть поле `task_id`, указывающее, к какой задаче он относится.
"""
        # Модифицируем Pydantic модель "на лету", чтобы требовать task_id
        class ClaimWithTaskId(ClaimList.model.__bases__[0]):
            task_id: str = Field(description="ID задачи, к которой относится это утверждение.")
        
        class ClaimListWithTaskId(BaseModel):
            claims: list[ClaimWithTaskId]

        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=ClaimListWithTaskId,
            budget_manager=self.budget_manager
        )
        return report.get('claims', [])

    def execute_batch(self, tasks: list[dict]) -> list:
        """Полный цикл работы 'рабочего' агента для пакета задач."""
        if not tasks:
            return []
            
        print(f"\n--- {self.__class__.__name__}: Приступаю к пакету из {len(tasks)} задач ---")
        
        # Шаг 1: Пакетная генерация запросов
        queries_by_task = self._decompose_tasks_batch(tasks)
        
        # Шаг 2: Выполнение всех поисковых запросов
        tasks_with_results = []
        for task in tasks:
            task_id = task['task_id']
            queries = queries_by_task.get(task_id, [])
            if not queries:
                continue
            
            raw_results = [self.search_agent.search(q) for q in queries]
            search_results_str = "\n".join([format_search_results_for_llm(r) for r in raw_results])
            tasks_with_results.append({
                "task": task,
                "search_results_str": search_results_str
            })

        if not tasks_with_results:
            print(f"!!! {self.__class__.__name__}: Поиск не дал результатов ни для одной задачи в пакете.")
            return []

        # Шаг 3: Пакетная генерация утверждений
        all_draft_claims = self._create_draft_claims_batch(tasks_with_results)
        
        print(f"--- {self.__class__.__name__}: Пакет обработан, всего сгенерировано {len(all_draft_claims)} сырых утверждений. ---")
        return all_draft_claims