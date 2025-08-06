# agents/expert_team.py
import json
from pydantic import BaseModel, Field 
from typing import List, Literal, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.search_agent import SearchAgent
from utils.helpers import format_search_results_for_llm

# --- PYDANTIC СХЕМЫ ДЛЯ ЭКСПЕРТНОЙ КОМАНДЫ ---

class SearchQueries(BaseModel):
    """Описывает список поисковых запросов."""
    queries: List[str] = Field(description="Список из 4-6 конкретных и разнообразных поисковых запросов на русском языке.")

class Claim(BaseModel):
    """Описывает одно конкретное, верифицируемое 'Утверждение' (Claim)."""
    claim_id: str = Field(description="Короткий, уникальный и информативный ID на английском (например, 'websoft_customization_weakness').")
    statement: str = Field(description="Четкий вывод или факт, сформулированный как утверждение.")
    value: str = Field(description="Конкретное значение (цифра, текст, список), подтверждающее утверждение.")
    source_link: str = Field(description="Прямая ссылка на самый релевантный источник, откуда взята информация.")
    source_quote: str = Field(description="Прямая цитата из источника, которая доказывает утверждение.")
    confidence_score: float = Field(description="Оценка уверенности в достоверности источника от 0.0 до 1.0.")
    status: Literal["UNVERIFIED", "VERIFIED"] = Field(description="Статус утверждения. На этапе создания всегда 'UNVERIFIED'.")

class ClaimList(BaseModel):
    """Описывает список 'Утверждений'."""
    claims: List[Claim]

class AuditReport(BaseModel):
    """Описывает отчет аудитора с перечнем уязвимостей для каждого утверждения."""
    vulnerabilities: Dict[str, List[str]] = Field(description="Словарь, где ключ - это claim_id, а значение - список найденных уязвимостей (текстовых описаний).")


class ExpertTeam:
    """
    Управляет командой экспертов. Получает задачу и ОБЩИЙ КОНТЕКСТ,
    проводит исследование, аудит и возвращает список верифицированных "Утверждений".
    """
    def __init__(self, llms: dict, search_agent: SearchAgent):
        self.llms = llms
        self.search_agent = search_agent
        print("-> Команда Экспертов сформирована и использует Pydantic-парсеры.")

    def _get_llm_for_expert(self, assignee: str) -> ChatGoogleGenerativeAI:
        """Выбирает модель в зависимости от роли эксперта."""
        return self.llms.get("expert_flash", self.llms["expert_lite"])

    def _invoke_llm_for_json(self, llm: ChatGoogleGenerativeAI, prompt: str, pydantic_schema: BaseModel) -> dict:
        """Надежный метод для вызова LLM с гарантированным JSON-ответом."""
        parser = PydanticOutputParser(pydantic_object=pydantic_schema)
        prompt_with_format_instructions = f"{prompt}\n\n{parser.get_format_instructions()}"
        
        try:
            response = llm.invoke(prompt_with_format_instructions)
            parsed_object = parser.parse(response.content)
            return parsed_object.model_dump()
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА LLM/Парсера в ExpertTeam: {e}")
            return {}

    def execute_task(self, task: dict, world_model_context: dict) -> list:
        """Основной метод, запускающий полный цикл работы над одной задачей."""
        assignee = task['assignee']
        description = task['description']
        goal = task['goal']
        print(f"\n--- Эксперт {assignee}: Приступаю к задаче '{description}' ---")

        try:
            # 1. Декомпозиция задачи
            queries_dict = self._decompose_task(assignee, description, goal, world_model_context)
            search_queries = queries_dict.get('queries', [])
            if not search_queries: return []

            # 2. Поиск и форматирование результатов
            raw_results = [self.search_agent.search(q) for q in search_queries]
            search_results_str = "\n".join([format_search_results_for_llm(r) for r in raw_results])
            
            if not search_results_str.strip() or "Поиск не дал результатов" in search_results_str:
                print(f"!!! Эксперт {assignee}: Поиск не дал результатов. Задача не может быть выполнена.")
                return []

            # 3. Написание черновика "Утверждений"
            draft_claims_dict = self._create_draft_claims(assignee, description, goal, search_results_str, world_model_context)
            draft_claims = draft_claims_dict.get('claims', [])
            if not draft_claims: return []

            # 4. Аудит
            vulnerabilities_dict = self._audit_claims(draft_claims, world_model_context)
            vulnerabilities = vulnerabilities_dict.get('vulnerabilities', {})
            
            # 5. Финализация
            final_claims_dict = self._finalize_claims(assignee, description, search_results_str, draft_claims, vulnerabilities, world_model_context)
            final_claims = final_claims_dict.get('claims', [])
            if not final_claims: return []

            print(f"--- Эксперт {assignee}: Задача выполнена, сгенерировано {len(final_claims)} верифицированных утверждений. ---")
            return final_claims

        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА в ExpertTeam при выполнении задачи '{description}': {e}")
            return []

    def _decompose_task(self, assignee: str, description: str, goal: str, context: dict) -> dict:
        """Шаг 1: Генерирует поисковые запросы."""
        print(f"   [Эксперт {assignee}] Шаг 1/5: Генерирую поисковые запросы...")
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}
**КОНТЕКСТ:**
Ты - ассистент-исследователь для эксперта '{assignee}'.
Цель твоего эксперта: {goal}
Текущая задача эксперта: {description}
**ТВОЯ ЗАДАЧА:**
Сгенерируй от 4 до 6 максимально конкретных и разнообразных поисковых запросов на русском языке, которые помогут эксперту найти ДОКАЗАТЕЛЬСТВА и ФАКТЫ для выполнения его задачи.
Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме."""
        llm = self._get_llm_for_expert(assignee)
        queries = self._invoke_llm_for_json(llm, prompt, SearchQueries)
        if queries.get('queries'):
            print(f"   [Эксперт {assignee}] -> Поисковые запросы сгенерированы: {queries['queries']}")
        else:
            print(f"!!! Эксперт {assignee}: Не удалось сгенерировать поисковые запросы.")
        return queries

    def _create_draft_claims(self, assignee: str, description: str, goal: str, search_results: str, context: dict) -> dict:
        """Шаг 2: Создает черновой список "Утверждений"."""
        print(f"   [Эксперт {assignee}] Шаг 2/5: Создаю черновик утверждений...")
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}
**ТВОЯ РОЛЬ И ЗАДАЧА:**
Ты - {assignee}. Твоя цель - {goal}.
Твоя текущая задача - проанализировать результаты поиска по теме '{description}' и сформулировать несколько ключевых "Утверждений" (Claims).
Каждое утверждение должно быть максимально конкретным, основанным на данных и РЕЛЕВАНТНЫМ для ОБЩЕЙ МИССИИ ПРОЕКТА.
**ПРАВИЛА:**
- Статус каждого утверждения должен быть 'UNVERIFIED'.
- Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
**РЕЗУЛЬТАТЫ ПОИСКА ДЛЯ АНАЛИЗА:**
---
{search_results}
---"""
        llm = self._get_llm_for_expert(assignee)
        claims = self._invoke_llm_for_json(llm, prompt, ClaimList)
        if claims.get('claims'):
            print(f"   [Эксперт {assignee}] -> Создан черновик из {len(claims['claims'])} утверждений.")
        else:
            print(f"!!! Эксперт {assignee}: Не удалось создать черновик утверждений.")
        return claims

    def _audit_claims(self, claims: list, context: dict) -> dict:
        """Шаг 3: "Враждебный Аудитор" проверяет утверждения."""
        print(f"   [Аудитор] Шаг 3/5: Провожу аудит {len(claims)} утверждений...")
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}
**ТВОЯ РОЛЬ И ЗАДАЧА:**
Твоя Роль: "Враждебный Аудитор". Ты не доверяешь ничему.
Твоя Задача: Тебе предоставлен список "Утверждений", сделанных другим AI. Твоя цель - найти в них логические ошибки, слабые места или недостаток доказательств, особенно те, которые могут ввести в заблуждение при достижении ОБЩЕЙ МИССИИ ПРОЕКТА.
Для каждого `claim_id` верни список текстовых "уязвимостей". Если уязвимостей нет, верни пустой список для этого `claim_id`.
**УТВЕРЖДЕНИЯ ДЛЯ АУДИТА:**
---
{json.dumps(claims, ensure_ascii=False, indent=2)}
---
Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме."""
        auditor_llm = self.llms["expert_flash"] # Аудитор всегда "умный"
        vulnerabilities = self._invoke_llm_for_json(auditor_llm, prompt, AuditReport)
        if vulnerabilities.get('vulnerabilities'):
            print(f"   [Аудитор] -> Проверка завершена.")
        else:
            print(f"!!! Аудитор: Не удалось провести аудит.")
        return vulnerabilities

    def _finalize_claims(self, assignee: str, description: str, search_results: str, draft_claims: list, vulnerabilities: dict, context: dict) -> dict:
        """Шаг 4: Эксперт дорабатывает утверждения с учетом критики."""
        print(f"   [Эксперт {assignee}] Шаг 4/5: Финализирую утверждения с учетом аудита...")
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}
**ТВОЯ РОЛЬ И ЗАДАЧА:**
Ты - {assignee}.
Твоя Задача: Пересмотреть свой черновик "Утверждений" по теме '{description}' с учетом отчета от "Враждебного Аудитора".
Исправь свои утверждения, уточни формулировки и, что **ОЧЕНЬ ВАЖНО**, скорректируй `confidence_score` на основе критики (например, если источник рекламный, понизь уверенность). Твои финальные утверждения должны быть максимально точными и полезными для ОБЩЕЙ МИССИИ ПРОЕКТА.
**ОРИГИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ПОИСКА:**
---
{search_results}
---
**ТВОЙ ЧЕРНОВИК УТВЕРЖДЕНИЙ:**
---
{json.dumps(draft_claims, ensure_ascii=False, indent=2)}
---
**ОТЧЕТ АУДИТОРА (УЯЗВИМОСТИ):**
---
{json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}
---
Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме. Все утверждения должны иметь статус 'UNVERIFIED'."""
        llm = self._get_llm_for_expert(assignee)
        final_claims_dict = self._invoke_llm_for_json(llm, prompt, ClaimList)
        
        if final_claims_dict.get('claims'):
            print(f"   [Эксперт {assignee}] -> Утверждения финализированы.")
            # Шаг 5: Присваиваем статус VERIFIED
            for claim in final_claims_dict['claims']:
                claim['status'] = 'VERIFIED'
            print(f"   [Эксперт {assignee}] Шаг 5/5: Статус {len(final_claims_dict['claims'])} утверждений обновлен на VERIFIED.")
            return final_claims_dict
        else:
            print(f"!!! Эксперт {assignee}: Не удалось финализировать утверждения.")
            return {}