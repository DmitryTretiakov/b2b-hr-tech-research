# agents/expert_team.py
import json
import os
import time
import uuid
from pydantic import BaseModel, Field 
from typing import List, Literal, Dict
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from agents.search_agent import SearchAgent
from utils.helpers import format_search_results_for_llm
from google.api_core.exceptions import ResourceExhausted
from core.world_model import WorldModel
from utils.helpers import invoke_llm_for_json_with_retry, read_system_logs, validate_artifact_citations
from core.budget_manager import APIBudgetManager

# --- PYDANTIC СХЕМЫ ДЛЯ ЭКСПЕРТНОЙ КОМАНДЫ ---

class SearchQueries(BaseModel):
    """Описывает список поисковых запросов."""
    queries: List[str] = Field(description="Список из 4-6 конкретных и разнообразных поисковых запросов на русском языке.")

class BatchVulnerabilityReport(BaseModel):
    """Пакетный отчет аудитора с перечнем уязвимостей для каждого утверждения."""
    vulnerabilities: Dict[str, List[str]] = Field(description="Словарь, где ключ - это claim_id, а значение - список найденных уязвимостей. Если уязвимостей для claim_id нет, его можно не включать в словарь или оставить с пустым списком.")

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

class FinancialReport(BaseModel):
    """Финансовый отчет, содержащий смету и прогноз в виде Markdown."""
    report_title: str = Field(description="Заголовок отчета, например 'Смета на MVP и Прогноз Прибыли на 1-й Год'.")
    markdown_content: str = Field(description="Полностью готовая Markdown-таблица и текстовые выводы.")

class ProductBrief(BaseModel):
    """Продуктовый бриф, содержащий описание и User Stories."""
    product_description: str = Field(description="Детальное описание продукта 'Карьерный Навигатор'.")
    user_stories: List[str] = Field(description="Список User Stories в формате 'Как <роль>, я хочу <действие>, чтобы <ценность>'.")

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
    def __init__(self, llms: dict, search_agent: SearchAgent, budget_manager: APIBudgetManager):
        self.llms = llms
        self.search_agent = search_agent
        self.budget_manager = budget_manager
        print("-> Команда Экспертов сформирована и использует Pydantic-парсеры.")

    def _execute_arbitration_task(self, task: dict, world_model: WorldModel) -> None:
        """
        Выполняет задачу по разрешению конфликта между двумя утверждениями.
        """
        print(f"   [Арбитр] -> Приступаю к разрешению конфликта: {task['description']}")

        if 'conflict_details' not in task or 'claim_ids' not in task['conflict_details'] or len(task['conflict_details']['claim_ids']) != 2:
            print(f"   [Арбитр] !!! КРИТИЧЕСКАЯ ОШИБКА: Задача '{task['task_id']}' назначена Арбитру, но не содержит корректного поля 'conflict_details'. Задача создана некорректно Стратегом.")
            world_model.update_task_status(task['task_id'], 'FAILED')
            return
        
        # --- 1. Извлечение данных о конфликте ---
        try:
            claim_ids = task['conflict_details']['claim_ids']
            claim_id_A = claim_ids[0]
            claim_id_B = claim_ids[1]
            
            kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
            claim_A = kb[claim_id_A]
            claim_B = kb[claim_id_B]
        except (KeyError, IndexError) as e:
            print(f"   [Арбитр] !!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось извлечь ID из поля 'conflict_details'. Ошибка: {e}")
            world_model.update_task_status(task['task_id'], 'FAILED')
            return

        # --- 2. Инициализация Gemma для этой задачи ---
        arbiter_llm = self._get_llm_for_task('SPECIALIST')

        # --- 3. Шаг 1 ReAct: Генерация поискового запроса ---
        prompt_step1 = f"""**ТВОЯ РОЛЬ:** AI-Арбитр, решающий конфликты.
**КОНФЛИКТ:** Тебе даны два противоречащих друг другу утверждения.
- Утверждение А (ID: {claim_A['claim_id']}): "{claim_A['statement']}" (Источник: {claim_A['source_link']})
- Утверждение Б (ID: {claim_B['claim_id']}): "{claim_B['statement']}" (Источник: {claim_B['source_link']})

**ТВОЯ ЗАДАЧА (Шаг 1):**
Сформулируй ОДИН, максимально точный и конкретный поисковый запрос на русском языке, который поможет найти авторитетный источник (например, официальный сайт, крупное СМИ, документацию) и однозначно определить, какое из утверждений верно.

Верни результат в виде JSON.
"""
        query_report = invoke_llm_for_json_with_retry(
            main_llm=arbiter_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt_step1,
            pydantic_schema=ArbitrationSearchQuery,
            budget_manager=self.budget_manager
        )
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
        final_report = invoke_llm_for_json_with_retry(
            main_llm=arbiter_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt_step2,
            pydantic_schema=ArbitrationReport,
            budget_manager=self.budget_manager
        )

        # --- 5. Обновление Базы Знаний на основе вердикта ---
        if not final_report or 'winning_claim_id' not in final_report:
            print("   [Арбитр] !!! Не удалось вынести финальное решение. Задача провалена.")
            world_model.update_task_status(task['task_id'], 'FAILED')
            return

        winner_id = final_report['winning_claim_id']
        loser_id = final_report['losing_claim_id']
        
        claims_to_update = []
        if winner_id in kb:
            kb[winner_id]['status'] = 'VERIFIED'
            claims_to_update.append(kb[winner_id])
            print(f"   [Арбитр] Утверждение {winner_id} подтверждено.")

        if loser_id in kb:
            kb[loser_id]['status'] = 'DEPRECATED'
            claims_to_update.append(kb[loser_id])
            print(f"   [Арбитр] Утверждение {loser_id} признано устаревшим.")
        
        if claims_to_update:
            world_model.add_claims_to_kb(claims_to_update)

        print("   [Арбитр] <- Конфликт успешно разрешен.")
        world_model.update_task_status(task['task_id'], 'COMPLETED')

    def _get_llm_for_task(self, task_type: Literal['ROUTINE', 'NLI', 'AUDIT', 'SPECIALIST']) -> ChatGoogleGenerativeAI:
        """
        Единый диспетчер моделей. Выбирает LLM на основе типа задачи и доступного бюджета.
        """
        if task_type == 'NLI':
            gemma_model_name = "models/gemma-3-27b-it"
            if self.budget_manager.can_i_spend(gemma_model_name):
                return self.llms["source_auditor"]

            raise ResourceExhausted("All models suitable for NLI have reached their daily budget limit.")
        elif task_type == 'AUDIT':
            flash_model_name = "models/gemini-2.5-flash"
            if self.budget_manager.can_i_spend(flash_model_name):
                return self.llms["expert_flash"]
            gemma_model_name = "models/gemma-3-27b-it"
            if self.budget_manager.can_i_spend(gemma_model_name):
                return self.llms["source_auditor"]
            raise ResourceExhausted("All models for AUDIT have reached their daily budget limit.")
        elif task_type == 'ROUTINE':
            lite_model_name = "models/gemini-2.5-flash-lite"
            if self.budget_manager.can_i_spend(lite_model_name):
                return self.llms["expert_lite"]
            flash_model_name = "models/gemini-2.5-flash"
            if self.budget_manager.can_i_spend(flash_model_name):
                return self.llms["expert_flash"]
            gemma_model_name = "models/gemma-3-27b-it"
            if self.budget_manager.can_i_spend(gemma_model_name):
                return self.llms["source_auditor"]
            raise ResourceExhausted("All models for ROUTINE have reached their daily budget limit.")
        elif task_type == 'SPECIALIST':
            gemma_model_name = "models/gemma-3-27b-it"
            if self.budget_manager.can_i_spend(gemma_model_name):
                return self.llms["source_auditor"]
            raise ResourceExhausted("All models for SPECIALIST have reached their daily budget limit.")
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _batch_audit_sources(self, urls: List[str]) -> Dict[str, dict]:
        """
        Классифицирует пакет URL одним вызовом LLM и возвращает словарь с типами и коэффициентами доверия.
        """
        if not urls:
            return {}
            
        print(f"      [Пакетный Аудитор] -> Классифицирую {len(urls)} уникальных URL одним запросом...")
        # Используем быструю и дешевую модель для этой задачи
        auditor_llm = self._get_llm_for_task('ROUTINE')
        
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
        
        report = invoke_llm_for_json_with_retry(
            main_llm=auditor_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=BatchAuditReport,
            budget_manager=self.budget_manager
        )
        
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
        llm_for_nli = self._get_llm_for_task('NLI')

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
        
        report = invoke_llm_for_json_with_retry(
            main_llm=llm_for_nli,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=BatchNLIReport,
            budget_manager=self.budget_manager
        )
        
        # --- Дросселирование ПОСЛЕ вызова. 10 RPM -> 6с/запрос ---
        # Мы все еще делаем K вызовов, поэтому оставляем защиту от всплесков.
        print("         [Пакетный NLI Аудитор] Пауза 6 секунд для соблюдения лимита RPM...")
        time.sleep(6)

        return report.get("audit_results", [])
    
        
    def execute_task(self, task: dict, world_model: WorldModel) -> list:
        # --- 1. Определение общих переменных ---
        assignee = task.get('assignee')
        description = task.get('description', '')
        goal = task.get('goal', '')
        world_model_context = world_model.get_full_context()

        # --- 2. Главный диспетчер задач ---
        if assignee == 'ProductOwnerAgent':
            self._execute_arbitration_task(task, world_model)
            return []
        
        elif assignee == 'FinancialModelAgent':
            self._execute_financial_task(task, world_model)
            return []

        elif assignee == 'ProductManagerAgent':
            self._execute_product_task(task, world_model)
            return []
            
        elif assignee == 'System_Diagnostician':
            print(f"\n--- System Diagnostician: Анализирую логи системы ---")
            system_logs = read_system_logs(world_model.log_dir)
            
            prompt = f"""Проанализируй системные логи и определи любые аномалии, сбои или возможные проблемы.
Обрати особое внимание на паттерны ошибок, частоту сбоев и общую производительность системы.

СИСТЕМНЫЕ ЛОГИ:
{system_logs}

ЗАДАЧА:
{description}

ЦЕЛЬ:
{goal}

На основе этих данных создай список конкретных утверждений (claims) о состоянии системы."""
            
            try:
                claims = invoke_llm_for_json_with_retry(
                    main_llm=self.llms['expert_flash'],
                    sanitizer_llm=self.llms['source_auditor'],
                    prompt=prompt,
                    pydantic_schema=ClaimList,
                    budget_manager=self.budget_manager
                )
                
                if claims and 'claims' in claims:
                    for claim in claims['claims']:
                        claim['source_type'] = 'OFFICIAL_DOCS'  # Логи - официальный источник
                        claim['source_trust'] = 1.0  # Максимальное доверие к системным логам
                        claim['confidence_score'] = 1.0  # Максимальная уверенность для системных данных
                    return claims['claims']
                return []
            
            except Exception as e:
                print(f"!!! Ошибка при анализе системных логов: {e}")
                return []
        
        # --- 3. Стандартный конвейер для агентов-исследователей ---
        else:
            print(f"\n--- Эксперт {assignee}: Приступаю к задаче '{description}' ---")
            
            # Шаг 1: Декомпозиция задачи
            queries_dict = self._decompose_task(assignee, description, goal, world_model_context)
            search_queries = queries_dict.get('queries', [])
            if not search_queries: return []

            # Шаг 2: Поиск и форматирование результатов
            raw_results = [self.search_agent.search(q) for q in search_queries]
            # Проверяем, есть ли хоть один реальный результат во всех ответах
            has_any_real_results = any(res.get("items") for res in raw_results)
            if not has_any_real_results:
                print(f"!!! Эксперт {assignee}: Поиск по всем запросам не дал никаких органических результатов. Задача завершена без новых фактов.")
                return []
            search_results_str = "\n".join([format_search_results_for_llm(r) for r in raw_results])

            # Шаг 3: Написание черновика "Утверждений"
            draft_claims_dict = self._create_draft_claims(assignee, description, goal, search_results_str, world_model_context)
            draft_claims = draft_claims_dict.get('claims', [])
            if not draft_claims: return []

            # Шаг 3.5: Пакетный аудит источников
            print(f"   [Эксперт {assignee}] Шаг 3.5/6: Запускаю пакетный аудит источников...")
            unique_urls = list(set(claim['source_link'] for claim in draft_claims if 'source_link' in claim and claim['source_link']))
            batch_audit_report = self._batch_audit_sources(unique_urls)
            
            enriched_claims = []
            for claim in draft_claims:
                url = claim.get('source_link')
                audit_result = batch_audit_report.get(url, {"type": "UNKNOWN", "trust": 0.2})
                claim['source_type'] = audit_result['type']
                claim['source_trust'] = audit_result['trust']
                base_confidence = claim.get('confidence_score', 0.7) 
                claim['confidence_score'] = base_confidence * audit_result['trust']
                enriched_claims.append(claim)
            print(f"   [Эксперт {assignee}] -> Пакетный аудит и обогащение {len(enriched_claims)} утверждений завершены.")

            # Шаг 4: Аудит содержимого
            vulnerabilities_dict = self._batch_audit_claims(enriched_claims, world_model_context)
            if vulnerabilities_dict is None or 'vulnerabilities' not in vulnerabilities_dict:
                print(f"!!! Эксперт {assignee}: Процесс аудита провалился и не вернул структуру данных. Прерываю обработку.")
                return []
            vulnerabilities = vulnerabilities_dict['vulnerabilities']
            # Шаг 5: Финализация
            final_claims_dict = self._finalize_claims(assignee, description, search_results_str, enriched_claims, vulnerabilities, world_model_context)
            final_claims = final_claims_dict.get('claims', [])
            if not final_claims: return []

            # --- ШАГ 6: ИНТЕГРАЦИЯ С ПАКЕТНЫМ NLI-АУДИТОМ (ИЗМЕНЕННАЯ ЛОГИКА) ---
            print(f"   [Эксперт {assignee}] Шаг 6/6: Провожу финальную верификацию и интеграцию {len(final_claims)} утверждений...")
            
            claims_to_add_to_kb = []
            verified_claims_for_log = []
            knowledge_base = world_model_context['dynamic_knowledge']['knowledge_base']

            for new_claim in final_claims:
                is_conflicted = False
                similar_ids = world_model.semantic_index.find_similar_claim_ids(new_claim['statement'], top_k=5)
                
                if not similar_ids:
                    new_claim['status'] = 'VERIFIED'
                    claims_to_add_to_kb.append(new_claim)
                    verified_claims_for_log.append(new_claim)
                    continue

                existing_claims_to_check = [knowledge_base[an_id] for an_id in similar_ids if an_id in knowledge_base]
                
                if not existing_claims_to_check:
                    new_claim['status'] = 'VERIFIED'
                    claims_to_add_to_kb.append(new_claim)
                    verified_claims_for_log.append(new_claim)
                    continue

                audit_results = self._batch_nli_audit(new_claim, existing_claims_to_check)

                for result in audit_results:
                    if result.get("relationship") == "CONTRADICTS":
                        existing_claim_id = result['existing_claim_id']
                        print(f"!!! [Детектор Противоречий] ОБНАРУЖЕН КОНФЛИКТ между '{new_claim['claim_id']}' и '{existing_claim_id}'")
                        is_conflicted = True
                        
                        conflicted_claim = knowledge_base.get(existing_claim_id)
                        if conflicted_claim and conflicted_claim.get('status') != 'CONFLICTED':
                            conflicted_claim['status'] = 'CONFLICTED'
                            claims_to_add_to_kb.append(conflicted_claim)
                        
                        conflict_task = {
                            "task_id": f"conflict_{str(uuid.uuid4())[:8]}",
                            "assignee": "ProductOwnerAgent",
                            "description": f"Разрешить противоречие между утверждениями {new_claim['claim_id']} и {existing_claim_id}.",
                            "goal": "Обеспечить целостность Базы Знаний.",
                            "status": "PENDING", "retry_count": 0,
                            "conflict_details": {"claim_ids": [new_claim['claim_id'], existing_claim_id]}
                        }
                        world_model.add_task_to_plan(conflict_task) # Эта операция сама сохраняет состояние
                        break

                if is_conflicted:
                    new_claim['status'] = 'CONFLICTED'
                    claims_to_add_to_kb.append(new_claim)
                else:
                    new_claim['status'] = 'VERIFIED'
                    claims_to_add_to_kb.append(new_claim)
                    verified_claims_for_log.append(new_claim)

            # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: ОДИН ВЫЗОВ ДЛЯ ВСЕХ УТВЕРЖДЕНИЙ ---
            if claims_to_add_to_kb:
                print(f"   [Эксперт {assignee}] -> Пакетно добавляю {len(claims_to_add_to_kb)} утверждений в Базу Знаний...")
                world_model.add_claims_to_kb(claims_to_add_to_kb)
            
            print(f"--- Эксперт {assignee}: Задача выполнена, обработано {len(final_claims)} утверждений, из них {len(verified_claims_for_log)} непротиворечивых. ---")
            return verified_claims_for_log

    def _decompose_task(self, assignee: str, description: str, goal: str, context: dict) -> dict:
        """Шаг 1: Генерирует поисковые запросы."""


        role_prompt_addition = ""
        if assignee == 'Contrarian_Expert':
            role_prompt_addition = """
**ОСОБАЯ ИНСТРУКЦИЯ:** Ты "Адвокат Дьявола". Твоя цель — не подтвердить, а ОПРОВЕРГНУТЬ. Твои поисковые запросы должны быть сформулированы так, чтобы найти критику, негативные отзывы, провальные кейсы и альтернативные мнения. Например, вместо "преимущества X" ищи "недостатки X", "проблемы с X", "провал проекта X".
"""

        print(f"   [Эксперт {assignee}] Шаг 1/5: Генерирую поисковые запросы...")
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}
**ВАЖНЕЙШИЙ БИЗНЕС-КОНТЕКСТ (СТРОГО УЧИТЫВАТЬ):**
- **Локация Проекта:** {context['static_context']['project_location']['city']}, Россия.
- **Комментарий по Локации:** {context['static_context']['project_location']['comment']}

**КОНТЕКСТ ЗАДАЧИ:**
Ты - ассистент-исследователь для эксперта '{assignee}'.
Цель твоего эксперта: {goal}
Текущая задача эксперта: {description}
{role_prompt_addition}

**ТВОЯ ЗАДАЧА И ПРАВИЛА:**
1.  Сгенерируй от 4 до 6 максимально конкретных и разнообразных поисковых запросов на русском языке.
2.  **ПРАВИЛО ЛОКАЛИЗАЦИИ:** Ты ОБЯЗАН использовать географические уточнения (например, "в России", "в Томске") во всех запросах, где это уместно. Особенно это касается тем зарплат, стоимости аренды, рынка труда, законодательства и поиска местных компаний-конкурентов.
3.  **ИЗБЕГАЙ ГЛОБАЛЬНЫХ ДАННЫХ:** Не ищи информацию по рынку США или Европы, если она не имеет прямого отношения к рынку РФ. Твоя цель — собрать локально релевантные факты.

Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме."""
        llm = self._get_llm_for_task('ROUTINE')
        queries = invoke_llm_for_json_with_retry(
            main_llm=llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=SearchQueries,
            budget_manager=self.budget_manager
        )
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
- Сгенерируй НЕ БОЛЕЕ 7-10 самых важных и уникальных утверждений. Сосредоточься на качестве, а не на количестве.
- Статус каждого утверждения должен быть 'UNVERIFIED'.
- Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
**РЕЗУЛЬТАТЫ ПОИСКА ДЛЯ АНАЛИЗА:**
---
{search_results}
---"""
        llm = self._get_llm_for_task('ROUTINE')
        claims = invoke_llm_for_json_with_retry(
            main_llm=llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=ClaimList,
            budget_manager=self.budget_manager
        )
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
        auditor_llm = self._get_llm_for_task('AUDIT')
        vulnerabilities = invoke_llm_for_json_with_retry(
            main_llm=auditor_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=AuditReport,
            budget_manager=self.budget_manager
        )
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
        llm = self._get_llm_for_task('ROUTINE')
        final_claims_dict = invoke_llm_for_json_with_retry(
            main_llm=llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=ClaimList,
            budget_manager=self.budget_manager
        )
        
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
        
    def _execute_financial_task(self, task: dict, world_model: WorldModel):
        """Генерирует финансовую модель, используя Gemma."""
        print(f"   [FinancialModelAgent] -> Приступаю к задаче: {task['description']}")
        
        arbiter_llm = self._get_llm_for_task('AUDIT')
        
        # Используем RAG, чтобы дать агенту только релевантные факты
        relevant_claims = world_model.semantic_index.find_similar_claim_ids(task['description'], top_k=50)
        kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        context_kb = {claim_id: kb[claim_id] for claim_id in relevant_claims if claim_id in kb}

        prompt = f"""
**ТВОЯ РОЛЬ:** Финансовый аналитик в стартапе. Ты прагматичен и консервативен.Ты работаешь ТОЛЬКО с предоставленными данными. Ты не имеешь права ничего додумывать или использовать свои общие знания.
**ТВОЯ ЗАДАЧА:** "{task['description']}". На основе предоставленных фактов, создай смету на MVP и реалистичный прогноз прибыли на первый год.
**КАТЕГОРИЧЕСКИЕ ЗАПРЕТЫ (ПРАВИЛО №1):**
1.  **ЗАПРЕЩЕНО** использовать любую информацию, цифры или факты, которых нет в предоставленной ниже "БАЗЫ ЗНАНИЙ ДЛЯ АНАЛИЗА".
2.  **ЗАПРЕЩЕНО** производить любые математические расчеты (умножение, применение коэффициентов, вычисление среднего), если формула или коэффициент не предоставлены как отдельный факт в Базе Знаний.
3.  **ЗАПРЕЩЕНО** придумывать или "додумывать" любые цифры. Если данных для строки в смете нет, ты ОБЯЗАН написать в соответствующей ячейке "ДАННЫЕ ОТСУТСТВУЮТ".
+
**ПРОЧИЕ ПРАВИЛА:**
- Каждое число в твоих таблицах, которое взято из факта, ДОЛЖНО сопровождаться цитатой в формате [Источник: claim_id].
 
**ФАКТЫ ИЗ БАЗЫ ЗНАНИЙ ДЛЯ АНАЛИЗА:**

---
{json.dumps(context_kb, ensure_ascii=False, indent=2)}
---

**ИНСТРУКЦИИ:**
1.  **Проанализируй** факты о зарплатах, стоимости API, аренде и т.д.
2.  **Создай смету затрат на MVP** (6 месяцев разработки). Включи ФОТ, налоги, инфраструктуру.
3.  **Создай прогноз дохода на 1-й год.** Учитывай модели лицензирования, количество потенциальных клиентов.
4.  **Сделай вывод** о потенциальной рентабельности.
5.  **Сформируй результат** в виде ОДНОГО JSON-объекта с заголовком и полной Markdown-строкой, содержащей таблицы и выводы.
"""
        report = None
        print(f"      [FinancialModelAgent] Генерирую отчет с отказоустойчивым механизмом...")
        report = invoke_llm_for_json_with_retry(
            main_llm=arbiter_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=FinancialReport,
            budget_manager=self.budget_manager
        )
        
        # --- Проверка результата ПОСЛЕ цикла ---
        if report and 'markdown_content' in report:
            kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
            validation_result = validate_artifact_citations(report['markdown_content'], kb)
            
            if validation_result["is_valid"]:
                filename = "financial_model_mvp.md"
                world_model.save_artifact(filename, f"# {report['report_title']}\n\n{report['markdown_content']}")
                world_model.update_task_status(task['task_id'], 'COMPLETED')
            else:
                print(f"   [FinancialModelAgent] !!! Артефакт не прошел аудит цитат. Причина: {validation_result['reason']}")
                world_model.update_task_status(task['task_id'], 'FAILED')
        else:
            print(f"   [FinancialModelAgent] !!! Не удалось сгенерировать отчет после нескольких попыток.")
            world_model.update_task_status(task['task_id'], 'FAILED')

    def _execute_product_task(self, task: dict, world_model: WorldModel):
        """Генерирует продуктовый бриф, используя Gemma."""
        print(f"   [ProductManagerAgent] -> Приступаю к задаче: {task['description']}")
        
        arbiter_llm = self._get_llm_for_task('AUDIT')
        
        relevant_claims = world_model.semantic_index.find_similar_claim_ids(task['description'], top_k=70)
        kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        context_kb = {claim_id: kb[claim_id] for claim_id in relevant_claims if claim_id in kb}

        prompt = f"""**ТВОЯ РОЛЬ:** Опытный Владелец Продукта (Product Owner).
**ТВОЯ ЗАДАЧА:** "{task['description']}". На основе ВСЕХ имеющихся фактов, создай детальный продуктовый бриф.

**ФАКТЫ ИЗ БАЗЫ ЗНАНИЙ ДЛЯ АНАЛИЗА:**
---
{json.dumps(context_kb, ensure_ascii=False, indent=2)}
---

**ИНСТРУКЦИИ:**
1.  **Напиши детальное описание** продукта "Карьерный Навигатор", его цели и ключевые функции.
2.  **Сформулируй список User Stories** для MVP в формате "Как <роль>, я хочу <действие>, чтобы <ценность>".
3.  **Верни результат** в виде ОДНОГО JSON-объекта.
"""
        report = None
        print(f"      [ProductManagerAgent] Генерирую отчет с отказоустойчивым механизмом...")
        report = invoke_llm_for_json_with_retry(
            main_llm=arbiter_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=ProductBrief,
            budget_manager=self.budget_manager
        )

        # --- Проверка результата ПОСЛЕ цикла ---
        if report and 'product_description' in report and 'user_stories' in report:
            # Собираем весь текст для проверки
            full_content = f"# Описание\n{report['product_description']}\n\n# User Stories\n"
            full_content += "\n".join(report['user_stories'])

            kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
            validation_result = validate_artifact_citations(full_content, kb)

            if validation_result["is_valid"]:
                filename = "product_brief_mvp.md"
                content = f"# Описание Продукта: Карьерный Навигатор\n\n{report['product_description']}\n\n## User Stories для MVP\n\n"
                content += "\n".join([f"- {story}" for story in report['user_stories']])
                world_model.save_artifact(filename, content)
                world_model.update_task_status(task['task_id'], 'COMPLETED')
            else:
                print(f"   [ProductManagerAgent] !!! Артефакт не прошел аудит цитат. Причина: {validation_result['reason']}")
                world_model.update_task_status(task['task_id'], 'FAILED')
        else:
            print(f"   [ProductManagerAgent] !!! Не удалось сгенерировать отчет после нескольких попыток.")
            world_model.update_task_status(task['task_id'], 'FAILED')
    
    def _get_llm_for_nli(self) -> ChatGoogleGenerativeAI:
        """
        Выбирает наилучшую доступную модель СПЕЦИАЛЬНО для задачи NLI.
        Приоритет: Flash -> Gemma.
        """
        # Приоритет 1: Мощная модель, если есть бюджет
        gemma_model_name = "models/gemma-3-27b-it"
        
        if self.budget_manager.can_i_spend(gemma_model_name):
            print("   [NLI Диспетчер] Использую Gemma для NLI.")
            return self.llms["source_auditor"]

        # Приоритет 2: Резервная модель flash-lite
        flash_lite_model_name = "models/gemini-2.5-flash-lite"
        print(f"!!! [NLI Диспетчер] Бюджет для {flash_lite_model_name} исчерпан. Переключаюсь на {flash_lite_model_name} для NLI.")
        if self.budget_manager.can_i_spend(flash_lite_model_name):
            return self.llms["expert_lite"] 

        # Полный провал
        print("!!! КРИТИЧЕСКИЙ СБОЙ NLI: Все подходящие модели исчерпали дневной лимит.")
        raise ResourceExhausted("All models suitable for NLI have reached their daily budget limit.")
    
    def generate_contrarian_tasks(self, claims_to_challenge: list) -> list:
        """
        Детерминированно создает список задач для Contrarian_Expert на основе
        предоставленных утверждений. Не использует LLM.
        """
        print(f"   [ExpertTeam] -> Создаю контрарные задачи для {len(claims_to_challenge)} утверждений...")
        tasks = []
        for claim in claims_to_challenge:
            claim_id = claim.get('claim_id')
            statement = claim.get('statement')
            
            task_id = f"contrarian_task_{claim_id}"
            
            task = {
                "task_id": task_id,
                "assignee": "Contrarian_Expert",
                "description": f"Найти доказательства, которые опровергают, оспаривают или ставят под сомнение следующее утверждение: '{statement}'",
                "goal": f"Проверить на прочность ключевое утверждение '{claim_id}' для снижения предвзятости подтверждения.",
                "status": "PENDING",
                "retry_count": 0
            }
            tasks.append(task)
        
        print(f"   [ExpertTeam] <- Создано {len(tasks)} контрарных задач.")
        return tasks
    
    def _batch_audit_claims(self, claims_batch: list, context: dict) -> dict:
        """Проводит "враждебный аудит" для ПАКЕТА утверждений одним вызовом LLM."""
        if not claims_batch:
            return {}
        
        print(f"   [Пакетный Аудитор] -> Провожу аудит пакета из {len(claims_batch)} утверждений...")
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}
     **ТВОЯ РОЛЬ И ЗАДАЧА:**
     Твоя Роль: "Враждебный Аудитор". Ты не доверяешь ничему.
     Твоя Задача: Тебе предоставлен ПАКЕТ "Утверждений". Для КАЖДОГО `claim_id` из пакета найди логические ошибки, слабые места или недостаток доказательств.
**ТЫ ОБЯЗАН ПРОВЕРИТЬ КАЖДОЕ УТВЕРЖДЕНИЕ ПО СЛЕДУЮЩЕМУ ЧЕК-ЛИСТУ:**
1.  **Неоднозначность:** Содержит ли утверждение расплывчатые термины ("высокий", "много", "популярный") без конкретных цифр?
2.  **Отсутствие Контекста:** Указаны ли все необходимые условия (грейд специалиста, дата, город, до/после налогов), чтобы утверждение было истинным?
3.  **Сверхобобщение:** Не делается ли глобальный вывод (например, "для всей РФ") на основе локального источника (например, опрос в одном IT-сообществе)?
4.  **Предвзятость Источника:** Является ли источник (указан в `source_link` и `source_type`) коммерчески заинтересованной стороной (например, продавец решения)?
5.  **Устаревшие Данные:** Можно ли предположить, что информация (особенно о технологиях и зарплатах) могла устареть?
+
Для каждого `claim_id`, по которому ты нашел проблему хотя бы по одному пункту чек-листа, верни список конкретных уязвимостей. Если уязвимостей нет, не включай `claim_id` в ответ.
 
**ПАКЕТ УТВЕРЖДЕНИЙ ДЛЯ АУДИТА:**

---
{json.dumps(claims_batch, ensure_ascii=False, indent=2)}
---
Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
"""
        auditor_llm = self._get_llm_for_task('AUDIT') # Оставляем Gemini 2.5 Flash
        report = invoke_llm_for_json_with_retry(
            main_llm=auditor_llm,
            sanitizer_llm=self.llms['expert_lite'],
            prompt=prompt,
            pydantic_schema=BatchVulnerabilityReport,
            budget_manager=self.budget_manager
        )
        if report.get('vulnerabilities') is not None:
            print(f"   [Пакетный Аудитор] <- Проверка пакета завершена.")
        else:
            print(f"!!! Пакетный Аудитор: Не удалось провести аудит.")
        return report
