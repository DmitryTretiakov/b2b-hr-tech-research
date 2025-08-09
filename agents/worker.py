# agents/worker.py
import json
from typing import TYPE_CHECKING
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.search_agent import SearchAgent
from core.budget_manager import APIBudgetManager
from utils.helpers import format_search_results_for_llm, invoke_llm_for_json_with_retry
from agents.models import SearchQueries, ClaimList, BaseModel
from utils.prompt_engine import create_dynamic_prompt # ИМПОРТ

if TYPE_CHECKING:
    from core.world_model import WorldModel

class WorkerAgent:
    """
    Базовый класс для агентов, генерирующих сырые данные.
    Теперь использует динамические промпты для повышения релевантности.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, search_agent: SearchAgent, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.search_agent = search_agent
        self.budget_manager = budget_manager
        self.role_prompt = ""

    def _decompose_tasks_batch(self, dynamic_prompts: Dict[str, str]) -> dict:
        """Генерирует поисковые запросы для ПАКЕТА задач, используя динамические промпты."""
        print(f"   [{self.__class__.__name__}] Шаг 1/3: Генерирую сфокусированные поисковые запросы...")
        
        tasks_description = "\n".join([f"--- Task ID: {task_id} ---\n{prompt}\n" for task_id, prompt in dynamic_prompts.items()])

        prompt = f"""**ТВОЯ РОЛЬ:** Ты - ассистент-исследователь.
**ТВОЯ ЗАДАЧА:** Для КАЖДОЙ задачи из пакета, сгенерируй от 3 до 5 поисковых запросов, основываясь на ее УТОЧНЕННОМ ЗАДАНИИ.

**ПАКЕТ ЗАДАЧ ДЛЯ ДЕКОМПОЗИЦИИ:**
{tasks_description}
---
**ИНСТРУКЦИИ:**
Верни результат как ОДИН JSON-объект, где ключи - это `task_id`, а значения - списки строк-запросов.
"""
        response = self.llm.invoke(prompt)
        self.budget_manager.record_spend(self.llm.model)
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            print(f"!!! {self.__class__.__name__}: Не удалось распарсить JSON с поисковыми запросами.")
            return {}

    def _create_draft_claims_batch(self, tasks_with_search_results: list[dict], dynamic_prompts: Dict[str, str]) -> list[dict]:
        """Создает черновой список "Утверждений", используя динамические промпты."""
        print(f"   [{self.__class__.__name__}] Шаг 2/3: Создаю черновик утверждений...")
        
        formatted_input = []
        for item in tasks_with_search_results:
            task_id = item['task']['task_id']
            formatted_input.append(f"""
---
**Task ID:** {task_id}
**Инструкции для тебя:** {dynamic_prompts.get(task_id, "Нет инструкций")}
**Search Results:**
{item['search_results_str']}
---
""")
        
        prompt = f"""**ТВОЯ РОЛЬ:** Ты - {self.__class__.__name__}.
**ТВОЯ ЗАДАЧА:** Проанализируй результаты поиска для КАЖДОЙ из предоставленных задач и сгенерируй список "Утверждений" (Claims), строго следуя ИНСТРУКЦИЯМ.

**ПАКЕТ ЗАДАЧ И РЕЗУЛЬТАТОВ ПОИСКА:**
{''.join(formatted_input)}

**ИНСТРУКЦИИ ПО ВЫВОДУ:**
- Для каждой задачи сгенерируй НЕ БОЛЕЕ 5-7 самых релевантных утверждений.
- Статус каждого утверждения должен быть 'UNVERIFIED'.
- Верни результат как ОДИН JSON-объект со списком `claims`.
- В каждом объекте Claim должно быть поле `task_id`, указывающее, к какой задаче он относится.
"""
        class ClaimWithTaskId(ClaimList.model.__bases__[0]):
            task_id: str
        class ClaimListWithTaskId(BaseModel):
            claims: list[ClaimWithTaskId]

        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm, sanitizer_llm=self.sanitizer_llm,
            prompt=prompt, pydantic_schema=ClaimListWithTaskId, budget_manager=self.budget_manager
        )
        return report.get('claims', [])

    def execute_batch(self, tasks: list[dict], world_model: 'WorldModel') -> list:
        if not tasks: return []
            
        print(f"\n--- {self.__class__.__name__}: Приступаю к пакету из {len(tasks)} задач ---")
        
        # Шаг 0: Динамическая генерация промптов
        dynamic_prompts = {}
        kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        for task in tasks:
            relevant_ids = world_model.semantic_index.find_similar_claim_ids(task['description'], top_k=3)
            relevant_facts = [kb[fact_id] for fact_id in relevant_ids if fact_id in kb]
            dynamic_prompts[task['task_id']] = create_dynamic_prompt(self.role_prompt, task, relevant_facts)
        
        # Шаг 1: Пакетная генерация запросов
        queries_by_task = self._decompose_tasks_batch(dynamic_prompts)
        
        # Шаг 2: Выполнение всех поисковых запросов
        tasks_with_results = []
        for task in tasks:
            queries = queries_by_task.get(task['task_id'], [])
            if not queries: continue
            raw_results = [self.search_agent.search(q) for q in queries]
            search_results_str = "\n".join([format_search_results_for_llm(r) for r in raw_results])
            tasks_with_results.append({"task": task, "search_results_str": search_results_str})

        if not tasks_with_results: return []

        # Шаг 3: Пакетная генерация утверждений
        all_draft_claims = self._create_draft_claims_batch(tasks_with_results, dynamic_prompts)
        
        print(f"--- {self.__class__.__name__}: Пакет обработан, всего сгенерировано {len(all_draft_claims)} сырых утверждений. ---")
        return all_draft_claims