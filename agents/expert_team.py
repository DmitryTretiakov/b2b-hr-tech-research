# agents/expert_team.py
import json
import uuid
from pydantic import BaseModel, Field 
from typing import List, Literal, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.search_agent import SearchAgent
from utils.helpers import format_search_results_for_llm

from core.world_model import WorldModel

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
    status: Literal["UNVERIFIED", "VERIFIED", "CONFLICTED"] = Field(description="Статус утверждения. На этапе создания всегда 'UNVERIFIED'.")
    source_type: str = Field(default="UNKNOWN", description="Тип источника (например, OFFICIAL_DOCS, FORUM_POST).")
    source_trust: float = Field(default=0.2, description="Коэффициент доверия к самому ИСТОЧНИКУ, от 0.0 до 1.0.")

class ClaimList(BaseModel):
    """Описывает список 'Утверждений'."""
    claims: List[Claim]

class AuditReport(BaseModel):
    """Описывает отчет аудитора с перечнем уязвимостей для каждого утверждения."""
    vulnerabilities: Dict[str, List[str]] = Field(description="Словарь, где ключ - это claim_id, а значение - список найденных уязвимостей (текстовых описаний).")

class NLIReport(BaseModel):
    """Описывает результат анализа взаимосвязи двух утверждений."""
    relationship: Literal["CONTRADICTS", "SUPPORTS", "NEUTRAL"] = Field(description="Отношение второго утверждения к первому.")

SOURCE_TRUST_MULTIPLIERS = {
    "OFFICIAL_DOCS": 1.0,      # Официальная документация продукта или технологии
    "MAJOR_TECH_MEDIA": 0.9,   # Статья на крупном IT-ресурсе (Habr, CNews, TechCrunch)
    "ACADEMIC_PAPER": 0.8,     # Научная статья, публикация в рецензируемом журнале
    "COMPANY_BLOG": 0.7,       # Официальный блог компании-разработчика или консалтинговой фирмы
    "PRESS_RELEASE": 0.6,      # Официальный пресс-релиз
    "NEWS_ARTICLE": 0.5,       # Новостная статья в неспециализированном СМИ
    "MARKETING_CONTENT": 0.4,  # Рекламная статья, лендинг, брошюра
    "FORUM_POST": 0.3,         # Пост на форуме, Stack Overflow, Reddit
    "UNKNOWN": 0.2             # Неизвестный или неклассифицируемый источник
}

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
        
    def _are_claims_related(self, claim_A_text: str, claim_B_text: str) -> bool:
        """Использует быструю LLM для определения, говорят ли два утверждения об одном и том же."""
        print(f"      [Детектор Связи] -> Проверяю связь между утверждениями...")
        llm = self.llms["expert_lite"]
        prompt = f"""Утверждение А: "{claim_A_text}"
Утверждение Б: "{claim_B_text}"

Эти два утверждения касаются одного и того же ключевого объекта, концепции или показателя?
Ответь ТОЛЬКО "Да" или "Нет"."""
        try:
            response = llm.invoke(prompt)
            return "да" in response.content.lower()
        except Exception:
            return False # В случае ошибки считаем, что не связаны

    # --- НОВЫЙ МЕТОД ДЛЯ NLI-АУДИТА ---
    def _perform_nli_audit(self, claim_A: dict, claim_B: dict) -> str:
        """Проверяет два связанных утверждения на предмет противоречия."""
        print(f"      [NLI Аудитор] -> Найдена связь. Проверяю на противоречие...")
        llm = self.llms["expert_flash"] # Используем более мощную модель для NLI
        
        prompt = f"""Твоя задача - определить логическое отношение между двумя утверждениями.
Утверждение А (существующее в базе): "{claim_A['statement']}" (Значение: {claim_A['value']})
Утверждение Б (новое): "{claim_B['statement']}" (Значение: {claim_B['value']})

Противоречит ли Утверждение Б Утверждению А?
- CONTRADICTS: Если утверждения не могут быть истинными одновременно.
- SUPPORTS: Если одно утверждение поддерживает или подтверждает другое.
- NEUTRAL: Если утверждения о разном или не связаны логически.
"""
        report = self._invoke_llm_for_json(llm, prompt, NLIReport)
        relationship = report.get("relationship", "NEUTRAL")
        print(f"      [NLI Аудитор] <- Результат: {relationship}")
        return relationship
    
    def _audit_source(self, url: str) -> dict:
        """
        Классифицирует URL с помощью gemma-3-27b-it и возвращает тип и коэффициент доверия.
        """
        print(f"      [Аудитор Источников] -> Классифицирую URL: {url}")
        auditor_llm = self.llms["source_auditor"]
        
        source_types = list(SOURCE_TRUST_MULTIPLIERS.keys())
        
        prompt = f"""Твоя задача - классифицировать тип источника по его URL.
URL: "{url}"

Выбери ОДИН наиболее подходящий тип из этого списка: {source_types}

Верни в ответе ТОЛЬКО и ИСКЛЮЧИТЕЛЬНО название типа. Например: OFFICIAL_DOCS
"""
        try:
            response = auditor_llm.invoke(prompt)
            # Убираем лишние пробелы и возможные markdown-конструкции
            source_type = response.content.strip().replace("`", "")
            
            if source_type not in source_types:
                print(f"      [Аудитор Источников] !!! Внимание: Модель вернула невалидный тип '{source_type}'. Используется UNKNOWN.")
                source_type = "UNKNOWN"

            trust_multiplier = SOURCE_TRUST_MULTIPLIERS.get(source_type, 0.2)
            print(f"      [Аудитор Источников] <- URL классифицирован как {source_type} с доверием {trust_multiplier}.")
            return {"type": source_type, "trust": trust_multiplier}
        except Exception as e:
            print(f"      [Аудитор Источников] !!! КРИТИЧЕСКАЯ ОШИБКА при классификации URL: {e}")
            return {"type": "UNKNOWN", "trust": 0.2}
        
    def execute_task(self, task: dict, world_model: WorldModel) -> list:
        """Основной метод, запускающий полный цикл работы над одной задачей."""
        assignee = task['assignee']
        description = task['description']
        goal = task['goal']
        world_model_context = world_model.get_full_context()
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

            # --- НОВЫЙ ШАГ 3.5: АУДИТ ИСТОЧНИКОВ И ОБОГАЩЕНИЕ УТВЕРЖДЕНИЙ ---
            print(f"   [Эксперт {assignee}] Шаг 3.5/6: Провожу аудит источников для {len(draft_claims)} утверждений...")
            enriched_claims = []
            for claim in draft_claims:
                source_audit_results = self._audit_source(claim['source_link'])
                claim['source_type'] = source_audit_results['type']
                claim['source_trust'] = source_audit_results['trust']
                # Корректируем изначальную уверенность с учетом доверия к источнику
                claim['confidence_score'] *= source_audit_results['trust']
                enriched_claims.append(claim)
            print(f"   [Эксперт {assignee}] -> Аудит источников завершен.")

            # 4. Аудит
            vulnerabilities_dict = self._audit_claims(draft_claims, world_model_context)
            vulnerabilities = vulnerabilities_dict.get('vulnerabilities', {})
            
            # 5. Финализация
            final_claims_dict = self._finalize_claims(assignee, description, search_results_str, draft_claims, vulnerabilities, world_model_context)
            final_claims = final_claims_dict.get('claims', [])
            if not final_claims: return []

            # --- ИСПРАВЛЕННЫЙ ШАГ 6: ПОШТУЧНАЯ ВЕРИФИКАЦИЯ И ИНТЕГРАЦИЯ ---
            print(f"   [Эксперт {assignee}] Шаг 6/6: Провожу финальную верификацию и интеграцию {len(final_claims)} утверждений...")
            verified_claims_for_log = []
            knowledge_base = world_model_context['dynamic_knowledge']['knowledge_base']

            for new_claim in final_claims:
                is_conflicted = False
                # Проверяем новый claim на конфликт со всей текущей базой знаний
                for existing_claim_id, existing_claim in knowledge_base.items():
                    if new_claim['claim_id'] == existing_claim_id: continue

                    if self._are_claims_related(existing_claim['statement'], new_claim['statement']):
                        relationship = self._perform_nli_audit(existing_claim, new_claim)
                        if relationship == "CONTRADICTS":
                            print(f"!!! [Детектор Противоречий] ОБНАРУЖЕН КОНФЛИКТ между новым claim '{new_claim['claim_id']}' и существующим '{existing_claim_id}'")
                            is_conflicted = True
                            
                            existing_claim['status'] = 'CONFLICTED'
                            world_model.add_claims_to_kb(existing_claim)
                            
                            conflict_task = {
                                "task_id": f"conflict_{str(uuid.uuid4())[:8]}",
                                "assignee": "ProductOwnerAgent",
                                "description": f"Разрешить противоречие между утверждениями {new_claim['claim_id']} и {existing_claim_id}. Найди третий, решающий источник.",
                                "goal": "Обеспечить целостность Базы Знаний.",
                                "status": "PENDING", "retry_count": 0
                            }
                            world_model.add_task_to_plan(conflict_task)
                            break # Нашли конфликт, прекращаем проверку для этого new_claim

                # Решение о судьбе нового claim принимается здесь, ПОСЛЕ проверки всей базы
                if is_conflicted:
                    new_claim['status'] = 'CONFLICTED'
                    world_model.add_claims_to_kb(new_claim)
                else:
                    new_claim['status'] = 'VERIFIED'
                    world_model.add_claims_to_kb(new_claim)
                    verified_claims_for_log.append(new_claim)

            print(f"--- Эксперт {assignee}: Задача выполнена, интегрировано {len(verified_claims_for_log)} непротиворечивых утверждений. ---")
            return verified_claims_for_log
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