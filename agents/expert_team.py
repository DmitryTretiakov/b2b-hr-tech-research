# agents/expert_team.py
import json
import time
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
    status: Literal["UNVERIFIED", "VERIFIED", "CONFLICTED", "DEPRECATED"] = Field(description="Статус утверждения. На этапе создания всегда 'UNVERIFIED'.")
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
class SourceAuditResult(BaseModel):
    """Результат аудита для одного источника."""
    source_type: Literal[
        "OFFICIAL_DOCS", "MAJOR_TECH_MEDIA", "ACADEMIC_PAPER", 
        "COMPANY_BLOG", "PRESS_RELEASE", "NEWS_ARTICLE", 
        "MARKETING_CONTENT", "FORUM_POST", "UNKNOWN"
    ] = Field(description="Тип источника, выбранный из допустимого списка.")

class NLIResult(BaseModel):
    """Описывает результат сравнения одного нового утверждения с одним существующим."""
    existing_claim_id: str = Field(description="ID существующего утверждения, с которым проводилось сравнение.")
    relationship: Literal["CONTRADICTS", "SUPPORTS", "NEUTRAL"] = Field(description="Отношение нового утверждения к существующему.")

class BatchNLIReport(BaseModel):
    """Описывает полный отчет по пакетному сравнению одного нового утверждения с несколькими существующими."""
    audit_results: List[NLIResult] = Field(description="Список результатов сравнения для каждой пары.")

class BatchAuditReport(BaseModel):
    """Отчет аудитора по пакету источников."""
    audit_results: Dict[str, SourceAuditResult] = Field(description="Словарь, где ключ - это URL источника, а значение - результат его аудита.")

class ArbitrationSearchQuery(BaseModel):
    """Описывает поисковый запрос, который должен разрешить конфликт."""
    query: str = Field(description="Единственный, очень конкретный поисковый запрос на русском языке для нахождения решающего источника.")

class ArbitrationReport(BaseModel):
    """Описывает финальное решение Арбитра по конфликту."""
    reasoning: str = Field(description="Краткое и четкое объяснение, почему было принято такое решение, на основе нового источника.")
    winning_claim_id: str = Field(description="ID утверждения, которое было признано верным.")
    losing_claim_id: str = Field(description="ID утверждения, которое было признано ложным/устаревшим.")
    decisive_source_link: str = Field(description="Прямая ссылка на новый, решающий источник, который помог разрешить конфликт.")

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
    def _execute_arbitration_task(self, task: dict, world_model: WorldModel) -> None:
        """
        Выполняет задачу по разрешению конфликта между двумя утверждениями.
        """
        print(f"   [Арбитр] -> Приступаю к разрешению конфликта: {task['description']}")
        
        # --- 1. Извлечение данных о конфликте ---
        try:
            # "Разрешить противоречие между утверждениями claim_A и claim_B"
            parts = task['description'].split(' ')
            claim_id_A = parts[-3]
            claim_id_B = parts[-1]
            
            kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
            claim_A = kb[claim_id_A]
            claim_B = kb[claim_id_B]
        except (IndexError, KeyError) as e:
            print(f"   [Арбитр] !!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось распарсить ID конфликтующих утверждений из задачи. Ошибка: {e}")
            world_model.update_task_status(task['task_id'], 'FAILED')
            return

        # --- 2. Инициализация Gemma для этой задачи ---
        arbiter_llm = ChatGoogleGenerativeAI(
            model="models/gemma-3-27b-it",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )

        # --- 3. Шаг 1 ReAct: Генерация поискового запроса ---
        prompt_step1 = f"""**ТВОЯ РОЛЬ:** AI-Арбитр, решающий конфликты.
**КОНФЛИКТ:** Тебе даны два противоречащих друг другу утверждения.
- Утверждение А (ID: {claim_A['claim_id']}): "{claim_A['statement']}" (Источник: {claim_A['source_link']})
- Утверждение Б (ID: {claim_B['claim_id']}): "{claim_B['statement']}" (Источник: {claim_B['source_link']})

**ТВОЯ ЗАДАЧА (Шаг 1):**
Сформулируй ОДИН, максимально точный и конкретный поисковый запрос на русском языке, который поможет найти авторитетный источник (например, официальный сайт, крупное СМИ, документацию) и однозначно определить, какое из утверждений верно.

Верни результат в виде JSON.
"""
        query_report = self._invoke_llm_for_json(arbiter_llm, prompt_step1, ArbitrationSearchQuery)
        if not query_report or 'query' not in query_report:
            print("   [Арбитр] !!! Не удалось сгенерировать поисковый запрос. Задача провалена.")
            world_model.update_task_status(task['task_id'], 'FAILED')
            return
        
        search_query = query_report['query']
        print(f"   [Арбитр] Сгенерирован решающий запрос: '{search_query}'")

        # --- 4. Шаг 2 ReAct: Выполнение поиска и вынесение вердикта ---
        search_results = self.search_agent.search(search_query)
        formatted_results = format_search_results_for_llm(search_results)

        prompt_step2 = f"""**ТВОЯ РОЛЬ:** AI-Арбитр, решающий конфликты.
**КОНФЛИКТ:**
- Утверждение А (ID: {claim_A['claim_id']}): "{claim_A['statement']}"
- Утверждение Б (ID: {claim_B['claim_id']}): "{claim_B['statement']}"

**НОВЫЕ ДОКАЗАТЕЛЬСТВА:** Я выполнил для тебя поиск по запросу "{search_query}" и получил следующие результаты:
---
{formatted_results}
---

**ТВОЯ ЗАДАЧА (Шаг 2):**
1.  **Проанализируй** новые доказательства.
2.  **Прими финальное решение:** Какое из первоначальных утверждений (А или Б) подтверждается новыми доказательствами?
3.  **Заполни отчет** в формате JSON, указав ID победившего и проигравшего утверждения, ссылку на самый убедительный новый источник и краткое обоснование своего решения.
"""
        final_report = self._invoke_llm_for_json(arbiter_llm, prompt_step2, ArbitrationReport)

        # --- 5. Обновление Базы Знаний на основе вердикта ---
        if not final_report or 'winning_claim_id' not in final_report:
            print("   [Арбитр] !!! Не удалось вынести финальное решение. Задача провалена.")
            world_model.update_task_status(task['task_id'], 'FAILED')
            return

        winner_id = final_report['winning_claim_id']
        loser_id = final_report['losing_claim_id']
        
        # Обновляем статус победителя
        if winner_id in kb:
            kb[winner_id]['status'] = 'VERIFIED'
            world_model.add_claims_to_kb([kb[winner_id]]) # add_claims_to_kb ожидает список
            print(f"   [Арбитр] Утверждение {winner_id} подтверждено.")

        # Обновляем статус проигравшего
        if loser_id in kb:
            kb[loser_id]['status'] = 'DEPRECATED' # Новый статус для проигравших в споре
            world_model.add_claims_to_kb([kb[loser_id]])
            print(f"   [Арбитр] Утверждение {loser_id} признано устаревшим.")

        print("   [Арбитр] <- Конфликт успешно разрешен.")
        world_model.update_task_status(task['task_id'], 'COMPLETED')

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
    def _batch_audit_sources(self, urls: List[str]) -> Dict[str, dict]:
        """
        Классифицирует пакет URL одним вызовом LLM и возвращает словарь с типами и коэффициентами доверия.
        """
        if not urls:
            return {}
            
        print(f"      [Пакетный Аудитор] -> Классифицирую {len(urls)} уникальных URL одним запросом...")
        # Используем быструю и дешевую модель для этой задачи
        auditor_llm = self.llms["source_auditor"] 
        
        # Формируем описание типов и список URL для промпта
        source_types_list = list(SOURCE_TRUST_MULTIPLIERS.keys())
        urls_to_analyze = "\n".join([f'- "{url}"' for url in urls])

        prompt = f"""**ТВОЯ ЗАДАЧА:** Ты — AI-ассистент, классифицирующий веб-источники. Тебе предоставлен список URL. Для КАЖДОГО URL ты должен определить его тип.

**СПИСОК ДОПУСТИМЫХ ТИПОВ ИСТОЧНИКОВ:**
{source_types_list}

**URL ДЛЯ АНАЛИЗА:**
{urls_to_analyze}

**ИНСТРУКЦИЯ:** Верни результат в виде ОДНОГО JSON-объекта. Ключами в этом объекте должны быть предоставленные URL, а значениями — объекты с единственным полем 'source_type'. Если URL не соответствует ни одному типу или ссылка не работает, используй тип 'UNKNOWN'.
"""
        
        report = self._invoke_llm_for_json(auditor_llm, prompt, BatchAuditReport)
        
        processed_report = {}
        if report and 'audit_results' in report:
            for url, result_data in report['audit_results'].items():
                # Pydantic уже провалидировал тип, поэтому мы можем ему доверять
                source_type = result_data.get('source_type', 'UNKNOWN')
                trust = SOURCE_TRUST_MULTIPLIERS.get(source_type, 0.2)
                processed_report[url] = {"type": source_type, "trust": trust}
            print(f"      [Пакетный Аудитор] <- Успешно обработано {len(processed_report)} URL.")
        else:
            print(f"      [Пакетный Аудитор] !!! Пакетный аудит не вернул результатов. Для всех URL будет использован тип 'UNKNOWN'.")
            # Отказоустойчивость: если LLM не справился, присваиваем всем UNKNOWN
            for url in urls:
                processed_report[url] = {"type": "UNKNOWN", "trust": 0.2}

        return processed_report
    
    def _batch_nli_audit(self, new_claim: dict, existing_claims: List[dict]) -> List[dict]:
        """
        Проверяет одно новое утверждение против списка существующих одним пакетным вызовом LLM.
        """
        if not existing_claims:
            return []

        print(f"      [Пакетный NLI Аудитор] -> Сравниваю '{new_claim['claim_id']}' с {len(existing_claims)} кандидатами...")
        llm = self.llms["expert_flash"] # Надежная модель для NLI

        # Формируем текст существующих утверждений для промпта
        existing_claims_text = "\n".join(
            [f"- ID: {c['claim_id']}, Утверждение: \"{c['statement']}\" (Значение: {c['value']})" for c in existing_claims]
        )

        prompt = f"""**ТВОЯ ЗАДАЧА:** Ты — AI-аудитор, специализирующийся на поиске логических противоречий.
Твоя цель — определить логическое отношение между одним **НОВЫМ УТВЕРЖДЕНИЕМ** и списком **СУЩЕСТВУЮЩИХ УТВЕРЖДЕНИЙ**.

**НОВОЕ УТВЕРЖДЕНИЕ (Основное для сравнения):**
- ID: {new_claim['claim_id']}
- Утверждение: "{new_claim['statement']}" (Значение: {new_claim['value']})

**СУЩЕСТВУЮЩИЕ УТВЕРЖДЕНИЯ (Кандидаты для сравнения):**
{existing_claims_text}

**ИНСТРУКЦИИ:**
Для КАЖДОГО существующего утверждения из списка, определи его отношение к НОВОМУ утверждению.
Возможные отношения:
- **CONTRADICTS:** Если утверждения не могут быть истинными одновременно.
- **SUPPORTS:** Если одно утверждение поддерживает или подтверждает другое.
- **NEUTRAL:** Если утверждения о разном или не связаны логически.

Верни результат в виде ОДНОГО JSON-объекта, соответствующего предоставленной схеме.
"""
        
        report = self._invoke_llm_for_json(llm, prompt, BatchNLIReport)
        
        # --- Дросселирование ПОСЛЕ вызова. 10 RPM -> 6с/запрос ---
        # Мы все еще делаем K вызовов, поэтому оставляем защиту от всплесков.
        print("         [Пакетный NLI Аудитор] Пауза 6 секунд для соблюдения лимита RPM...")
        time.sleep(6)

        return report.get("audit_results", [])
    
        
    def execute_task(self, task: dict, world_model: WorldModel) -> list:
        if task.get('assignee') == 'ProductOwnerAgent':
            self._execute_arbitration_task(task, world_model)
            # Задачи арбитра не генерируют новых утверждений, они меняют старые.
            # Поэтому мы возвращаем пустой список.
            return []
        assignee = task['assignee']
        description = task['description']
        goal = task['goal']
        world_model_context = world_model.get_full_context()
        print(f"\n--- Эксперт {assignee}: Приступаю к задаче '{description}' ---")

        # Шаги 1-5 остаются без изменений, но без глобального try...except
        # 1. Декомпозиция задачи
        queries_dict = self._decompose_task(assignee, description, goal, world_model_context)
        search_queries = queries_dict.get('queries', [])
        if not search_queries: return []

        # 2. Поиск и форматирование результатов
        raw_results = [self.search_agent.search(q) for q in search_queries]
        search_results_str = "\n".join([format_search_results_for_llm(r) for r in raw_results])
        if not search_results_str.strip() or "Поиск не дал результатов" in search_results_str:
            print(f"!!! Эксперт {assignee}: Поиск не дал результатов.")
            return []

        # 3. Написание черновика "Утверждений"
        draft_claims_dict = self._create_draft_claims(assignee, description, goal, world_model_context)
        draft_claims = draft_claims_dict.get('claims', [])
        if not draft_claims: return []

        # 3.5 Пакетный аудит источников
        print(f"   [Эксперт {assignee}] Шаг 3.5/6: Запускаю пакетный аудит источников...")
        
        # 3.5.1: Собрать все уникальные URL из черновиков утверждений
        unique_urls = list(set(
            claim['source_link'] 
            for claim in draft_claims 
            if 'source_link' in claim and claim['source_link']
        ))
        
        # 3.5.2: Вызвать пакетный метод ОДИН РАЗ
        batch_audit_report = self._batch_audit_sources(unique_urls)
        
        # 3.5.3: Обогатить утверждения результатами пакетного аудита
        enriched_claims = []
        for claim in draft_claims:
            url = claim.get('source_link')
            
            # Получаем результат аудита из словаря, с безопасным фолбэком
            audit_result = batch_audit_report.get(url, {"type": "UNKNOWN", "trust": 0.2})
            
            claim['source_type'] = audit_result['type']
            claim['source_trust'] = audit_result['trust']
            
            # Пересчитываем confidence_score с учетом доверия к источнику
            # Используем get с дефолтным значением на случай, если LLM не сгенерировал confidence_score
            base_confidence = claim.get('confidence_score', 0.7) 
            claim['confidence_score'] = base_confidence * audit_result['trust']
            
            enriched_claims.append(claim)
            
        print(f"   [Эксперт {assignee}] -> Пакетный аудит и обогащение {len(enriched_claims)} утверждений завершены.")

        # 4. Аудит содержимого
        vulnerabilities_dict = self._audit_claims(enriched_claims, world_model_context)
        vulnerabilities = vulnerabilities_dict.get('vulnerabilities', {})
        
        # 5. Финализация
        final_claims_dict = self._finalize_claims(assignee, description, enriched_claims, vulnerabilities, world_model_context)
        final_claims = final_claims_dict.get('claims', [])
        if not final_claims: return []

        # --- ФИНАЛЬНЫЙ ШАГ 6: ИНТЕГРАЦИЯ С ИСПОЛЬЗОВАНИЕМ ПАКЕТНОГО NLI-АУДИТА ---
        print(f"   [Эксперт {assignee}] Шаг 6/6: Провожу финальную верификацию и интеграцию {len(final_claims)} утверждений...")
        verified_claims_for_log = []
        knowledge_base = world_model_context['dynamic_knowledge']['knowledge_base']

        for new_claim in final_claims:
            is_conflicted = False
            
            # 1. Найти семантически близких кандидатов (без изменений)
            similar_ids = world_model.semantic_index.find_similar_claim_ids(new_claim['statement'], top_k=5)
            
            if not similar_ids:
                # Если похожих нет, просто верифицируем утверждение
                new_claim['status'] = 'VERIFIED'
                world_model.add_claims_to_kb(new_claim)
                verified_claims_for_log.append(new_claim)
                continue

            # 2. Собрать полные данные по найденным кандидатам
            existing_claims_to_check = []
            for an_id in similar_ids:
                if an_id in knowledge_base:
                    existing_claims_to_check.append(knowledge_base[an_id])
            
            if not existing_claims_to_check:
                new_claim['status'] = 'VERIFIED'
                world_model.add_claims_to_kb(new_claim)
                verified_claims_for_log.append(new_claim)
                continue

            # 3. Выполнить ОДИН пакетный NLI-аудит для нового утверждения против всех кандидатов
            audit_results = self._batch_nli_audit(new_claim, existing_claims_to_check)

            # 4. Проанализировать результаты пакетного аудита
            for result in audit_results:
                if result.get("relationship") == "CONTRADICTS":
                    existing_claim_id = result['existing_claim_id']
                    print(f"!!! [Детектор Противоречий] ОБНАРУЖЕН КОНФЛИКТ между '{new_claim['claim_id']}' и '{existing_claim_id}'")
                    is_conflicted = True
                    
                    # Помечаем существующее утверждение как конфликтное
                    conflicted_claim = knowledge_base.get(existing_claim_id)
                    if conflicted_claim:
                        conflicted_claim['status'] = 'CONFLICTED'
                        world_model.add_claims_to_kb(conflicted_claim)
                    
                    # Создаем задачу на разрешение конфликта
                    conflict_task = {
                        "task_id": f"conflict_{str(uuid.uuid4())[:8]}",
                        "assignee": "ProductOwnerAgent",
                        "description": f"Разрешить противоречие между утверждениями {new_claim['claim_id']} и {existing_claim_id}. Найди третий, решающий источник.",
                        "goal": "Обеспечить целостность Базы Знаний.",
                        "status": "PENDING", "retry_count": 0
                    }
                    world_model.add_task_to_plan(conflict_task)
                    break # Одного противоречия достаточно, чтобы пометить новое утверждение как конфликтное

            # 5. Решить судьбу нового утверждения
            if is_conflicted:
                new_claim['status'] = 'CONFLICTED'
                world_model.add_claims_to_kb(new_claim)
            else:
                new_claim['status'] = 'VERIFIED'
                world_model.add_claims_to_kb(new_claim)
                verified_claims_for_log.append(new_claim)

        print(f"--- Эксперт {assignee}: Задача выполнена, интегрировано {len(verified_claims_for_log)} непротиворечивых утверждений. ---")
        return verified_claims_for_log

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