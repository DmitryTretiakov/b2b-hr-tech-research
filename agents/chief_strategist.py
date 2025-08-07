# agents/chief_strategist.py
import json
import os
from pydantic import BaseModel, Field 
from typing import List, Literal, TYPE_CHECKING 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser

# --- РЕШЕНИЕ ПРОБЛЕМЫ ЦИКЛИЧЕСКОЙ ЗАВИСИМОСТИ ТИПОВ ---
if TYPE_CHECKING:
    from core.world_model import WorldModel

# --- ОПРЕДЕЛЕНИЕ СТРУКТУРЫ ПЛАНА С ПОМОЩЬЮ PYDANTIC ---

class Task(BaseModel):
    """Описывает одну конкретную задачу в рамках фазы проекта."""
    task_id: str = Field(description="Уникальный идентификатор задачи, например 'task_001'. Должен быть уникальным во всем плане.")
    assignee: Literal['HR_Expert', 'Finance_Expert', 'Competitor_Expert', 'Tech_Expert', 'ProductOwnerAgent'] = Field(description="Эксперт, которому поручена задача.")
    description: str = Field(description="Четкое и краткое описание задачи для эксперта.")
    goal: str = Field(description="Бизнес-цель, на которую направлена эта задача. Что мы хотим узнать?")
    status: Literal['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'] = Field(description="Текущий статус задачи. Новые задачи всегда PENDING.")
    retry_count: int = Field(default=0, description="Счетчик повторных попыток выполнения задачи в случае сбоя API.")

class BatchRelevanceReport(BaseModel):
    relevant_claim_ids: List[str] = Field(description="Список ID всех утверждений, которые были сочтены коммерчески релевантными.")

class ValidationReport(BaseModel):
    """Описывает результат проверки артефакта на качество."""
    is_valid: bool = Field(description="True, если артефакт прошел все проверки, иначе False.")
    reasons: List[str] = Field(description="Список конкретных причин, по которым артефакт был признан невалидным. Пустой, если is_valid=True.")

class Phase(BaseModel):
    """Описывает одну фазу проекта, состоящую из нескольких задач."""
    phase_name: str = Field(description="Название фазы, например 'Phase 1: Глубокая Разведка Активов ТГУ'.")
    status: Literal['PENDING', 'IN_PROGRESS', 'COMPLETED'] = Field(description="Текущий статус фазы.")
    tasks: List[Task] = Field(description="Список задач для этой фазы.")

class RagQuerySet(BaseModel):
    """Набор специализированных запросов для RAG, по одному на каждый домен."""
    market_and_finance_query: str = Field(description="Сжатый поисковый запрос о рынке, финансах, ROI, бизнес-модели.")
    tech_and_assets_query: str = Field(description="Сжатый поисковый запрос о технологиях, стеке, сильных и слабых сторонах активов ТГУ.")
    competitor_query: str = Field(description="Сжатый поисковый запрос о конкурентах, их продуктах и позиционировании.")

class StrategicPlan(BaseModel):
    """Описывает полный стратегический план проекта."""
    main_goal_status: Literal['IN_PROGRESS', 'READY_FOR_FINAL_BRIEF', 'FAILED'] = Field(description="Общий статус всего проекта. IN_PROGRESS, пока идет работа.")
    phases: List[Phase] = Field(description="Список всех фаз проекта.")

# --- НОВАЯ PYDANTIC СХЕМА ДЛЯ ФИЛЬТРА РЕЛЕВАНТНОСТИ ---
class RelevanceCheck(BaseModel):
    """Описывает результат проверки утверждения на коммерческую релевантность."""
    is_relevant: bool = Field(description="True, если утверждение напрямую помогает ответить хотя бы на один из бизнес-вопросов, иначе False.")

class ChiefStrategist:
    """
    "Мозг" системы. Создает план, проводит рефлексию и пишет финальные отчеты.
    Работает с самой мощной моделью (gemini-pro).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI):
        self.llm = llm
        print("-> ChiefStrategist (на базе gemini-pro) готов к работе.")

    def _invoke_llm_for_json(self, prompt: str, pydantic_schema: BaseModel):
        """
        Новый, надежный метод для вызова LLM с гарантированным JSON-ответом.
        """
        print("   [Стратег] -> Вызов LLM для генерации структурированного JSON...")
        parser = PydanticOutputParser(pydantic_object=pydantic_schema)
        
        prompt_with_format_instructions = f"""{prompt}

{parser.get_format_instructions()}
"""
        try:
            response = self.llm.invoke(prompt_with_format_instructions)
            parsed_object = parser.parse(response.content)
            print("   [Стратег] <- Ответ LLM успешно получен и распарсен.")
            # Возвращаем как словарь для совместимости с остальным кодом
            return parsed_object.model_dump() 
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА LLM/Парсера: Не удалось сгенерировать или распарсить JSON. Ошибка: {e}")
            # Возвращаем пустой словарь, чтобы система могла обработать сбой
            return {}
        
    

    # --- НОВЫЙ ПРИВАТНЫЙ МЕТОД ДЛЯ ФИЛЬТРАЦИИ БАЗЫ ЗНАНИЙ ---
    def _filter_kb_by_relevance(self, knowledge_base: dict) -> dict:
        """
        Фильтрует всю базу знаний, оставляя только коммерчески релевантные утверждения.
        Теперь использует пакетную обработку.
        """
        return self._batch_filter_kb_by_relevance(knowledge_base)


    def _invoke_llm_for_text(self, prompt: str) -> str:
        """Простой вызов LLM для генерации текста."""
        print("   [Стратег] -> Вызов LLM для генерации текста...")
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА LLM: Вызов API провалился. Ошибка: {e}")
            return ""

    def create_strategic_plan(self, world_model_context: dict) -> dict:
        """
        Генерирует первоначальный план с использованием Pydantic.
        """
        print("   [Стратег] Шаг 1: Создаю первоначальный стратегический план...")
        
        prompt = f"""**ОБЩИЙ КОНТЕКСТ ПРОЕКТА:**
---
{json.dumps(world_model_context['static_context'], ensure_ascii=False, indent=2)}
---
**ТВОЯ РОЛЬ:** Ты - Главный Продуктовый Стратег с сильным коммерческим чутьем. Твоя первая задача - создать комплексный, пошаговый план исследования для принятия инвестиционного решения.
**КЛЮЧЕВОЙ ПРИНЦИП ПЛАНИРОВАНИЯ:** План должен быть сфокусирован на поиске коммерчески жизнеспособных гипотез. Каждая задача должна в конечном итоге помогать ответить на вопрос "Как мы на этом заработаем и почему это выгодно для ТГУ?".
**ТВОЯ ЗАДАЧА:** Проанализируй "ВХОДНОЙ БРИФ ДЛЯ ВЕРИФИКАЦИИ". Сгенерируй долгосрочный стратегический план из 3-4 логических фаз. **ПЕРВАЯ ФАЗА** должна быть посвящена **"Глубокой Разведке Активов ТГУ"**. Последующие фазы должны быть направлены на анализ рынка, конкурентов, разработку бизнес-кейса и MVP. Для каждой фазы создай список из 2-4 первоочередных, практически-ориентированных задач.
**ДОСТУПНЫЕ ЭКСПЕРТЫ:** `HR_Expert`, `Finance_Expert`, `Competitor_Expert`, `Tech_Expert`, `ProductOwnerAgent`.

Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
"""
        plan = self._invoke_llm_for_json(prompt, StrategicPlan)

        if plan and "phases" in plan:
            print("   [Стратег] Первоначальный план успешно сгенерирован.")
            return plan
        else:
            print("!!! Стратег: Не удалось сгенерировать первоначальный план. Используется план по умолчанию.")
            return {"main_goal_status": "FAILED", "phases": []}

    def reflect_and_update_plan(self, world_model: 'WorldModel') -> dict:
        """
        Проводит рефлексию, используя централизованный RAG-метод для получения контекста.
        """
        print("   [Стратег] Шаг X: Провожу интеллектуальную рефлексию...")
        
        full_context = world_model.get_full_context()

        # Шаг А: Осмысление ситуации (без изменений)
        situation_summary = self._summarize_situation(full_context)
        if not situation_summary:
            print("!!! Стратег: Не удалось осмыслить ситуацию. План не будет обновлен.")
            return full_context['dynamic_knowledge']['strategic_plan']

        # Шаг Б: Получение сбалансированного контекста с помощью единого RAG-метода
        # Для рефлексии нам не нужен очень широкий контекст.
        k_for_reflection = {
            "market_and_finance_query": 15,
            "tech_and_assets_query": 10,
            "competitor_query": 10
        }
        relevant_kb = self._get_balanced_rag_context(world_model, situation_summary, k_for_reflection)

        # Шаг В: Генерация обновленного плана
        updated_plan = self._generate_updated_plan(situation_summary, full_context, relevant_kb)

        if updated_plan and "phases" in updated_plan:
            print("   [Стратег] Рефлексия завершена. План обновлен.")
            return updated_plan
        else:
            print("!!! Стратег: Не удалось сгенерировать обновленный план. План не будет обновлен.")
            return full_context['dynamic_knowledge']['strategic_plan']
        
    def _generate_rag_queries(self, situation_summary: str) -> dict:
        """
        Декомпозирует общую мысль на несколько сфокусированных запросов для Multi-Query RAG.
        """
        print("      [Стратег.RAG] -> Декомпозирую общую мысль на сфокусированные запросы...")
        # Используем быструю модель для этой задачи
        query_gen_llm = ChatGoogleGenerativeAI(
            model="models/gemma-3-27b-it",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0
        )

        prompt = f"""**ТВОЯ ЗАДАЧА:** Ты — системный аналитик. Твоя цель — преобразовать общую аналитическую сводку в несколько конкретных, сжатых поисковых запросов для векторной базы данных.

**АНАЛИТИЧЕСКАЯ СВОДКА ОТ СТРАТЕГА:**
---
{situation_summary}
---

**ИНСТРУКЦИЯ:**
На основе этой сводки, сформулируй три очень коротких и сфокусированных запроса, по одному на каждый домен. Запросы должны отражать суть сводки в контексте каждого домена.

Верни результат в виде ОДНОГО JSON-объекта.
"""
        
        parser = PydanticOutputParser(pydantic_object=RagQuerySet)
        prompt_with_instructions = f"{prompt}\n\n{parser.get_format_instructions()}"
        
        try:
            response = query_gen_llm.invoke(prompt_with_instructions)
            parsed_object = parser.parse(response.content)
            report = parsed_object.model_dump()
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при генерации RAG-запросов: {e}")
            report = {}
        # --------------------------------------------------------------------

        return report
    
    def _summarize_situation(self, world_model_context: dict) -> str:
        """Шаг А рефлексии: Анализ и выводы в свободной форме."""
        print("      [Стратег.Рефлексия] Шаг А: Анализирую текущую ситуацию...")
        prompt = f"""**ТВОЯ РОЛЬ:** Главный Продуктовый Стратег.
**ТВОЯ ЗАДАЧА:** Проанализировать все имеющиеся данные по завершенной фазе исследования и написать краткую сводку (summary) в свободной форме.

**ПОЛНЫЙ КОНТЕКСТ И РЕЗУЛЬТАТЫ:**
---
{json.dumps(world_model_context, ensure_ascii=False, indent=2)}
---

**ТВОЙ МЫСЛИТЕЛЬНЫЙ ПРОЦЕСС:**
1.  **Проанализируй проваленные задачи (status: FAILED):** В чем причина? Критична ли потеря этой информации?
2.  **Оцени полноту Базы Знаний:** Достаточно ли собранных "Утверждений" для достижения целей завершенной фазы? Какие главные инсайты мы получили?
3.  **Сформулируй выводы:** Напиши 3-4 абзаца с ключевыми выводами и определи, готова ли команда переходить к следующей фазе или нужно провести дополнительное исследование.

Твоя аналитическая сводка:"""
        
        return self._invoke_llm_for_text(prompt)

    def _generate_updated_plan(self, situation_summary: str, full_context: dict, relevant_kb: dict) -> dict:
        """Шаг Б рефлексии: Превращение выводов в конкретный JSON-план с помощью Pydantic."""
        print("      [Стратег.Рефлексия] Шаг Б: Превращаю выводы в конкретный план...")
        
        # Создаем "облегченный" контекст для передачи в LLM
        lean_context_for_llm = {
            "static_context": full_context['static_context'],
            "dynamic_knowledge": {
                # Передаем оригинальный план, чтобы модель знала, что обновлять
                "strategic_plan": full_context['dynamic_knowledge']['strategic_plan'],
                # Передаем ТОЛЬКО релевантный срез из Базы Знаний
                "relevant_knowledge_base_slice": relevant_kb 
            }
        }
        lean_context_str = json.dumps(lean_context_for_llm, ensure_ascii=False, indent=2)

        prompt = f"""**ТВОЯ РОЛЬ:** Ассистент-планировщик.
**ТВОЯ ЗАДАЧА:** Тебе предоставлены выводы от Главного Стратега и текущий план. Твоя задача - обновить план в соответствии с выводами, опираясь на **релевантные факты из Базы Знаний**.

**ВЫВОДЫ ГЛАВНОГО СТРАТЕГА:**
---
{situation_summary}
---

**ОБЛЕГЧЕННЫЙ КОНТЕКСТ (ТЕКУЩИЙ ПЛАН И РЕЛЕВАНТНЫЕ ФАКТЫ):**
---
{lean_context_str}
---

**ИНСТРУКЦИИ ПО ОБНОВЛЕНИЮ:**
1.  Заверши текущую активную фазу (измени ее статус на "COMPLETED").
2.  Если Стратег решил, что нужно дополнительное исследование, добавь новые задачи в **текущую** или **следующую** фазу.
3.  Если Стратег решил, что все готово, начни следующую фазу или, если это последняя, измени `main_goal_status` на `READY_FOR_FINAL_BRIEF`.
4.  Верни **полностью обновленный объект стратегического плана**. Убедись, что новые `task_id` уникальны и не повторяют существующие.

Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
"""
        return self._invoke_llm_for_json(prompt, StrategicPlan)

    def write_executive_summary(self, world_model: 'WorldModel', feedback: str = None) -> str:
        """Пишет короткую аналитическую записку, используя RAG-контекст."""
        print("   [Стратег] Финальный Шаг (1/2): Пишу краткую аналитическую записку (RAG-подход)...")
        
        # Для финального отчета нам нужен максимально полный и сбалансированный контекст.
        k_for_summary = {
            "market_and_finance_query": 40,
            "tech_and_assets_query": 30,
            "competitor_query": 20
        }
        # В качестве "мысли" для RAG используем главную цель проекта.
        main_goal_as_query = world_model.get_full_context()['static_context']['main_goal']
        relevant_kb = self._get_balanced_rag_context(world_model, main_goal_as_query, k_for_summary)

        # Создаем облегченный контекст для промпта
        lean_context = {"relevant_knowledge_base": relevant_kb}

        feedback_section = ""
        if feedback:
            feedback_section = f"""... (код feedback_section без изменений) ..."""

        prompt = f"""**ТВОЯ РОЛЬ:** ... (основной промпт без изменений) ...
**КОНТЕКСТ И БАЗА ЗНАНИЙ (ТОЛЬКО РЕЛЕВАНТНЫЕ ФАКТЫ):**
---
{json.dumps(lean_context, ensure_ascii=False, indent=2)}
---
... (остальная часть промпта) ...
{feedback_section}
Твоя финальная аналитическая записка:"""
        return self._invoke_llm_for_text(prompt)

    def write_extended_brief(self, world_model: 'WorldModel', feedback: str = None) -> str:
        """Пишет подробный обзор, используя RAG-контекст."""
        print("   [Стратег] Финальный Шаг (2/2): Пишу подробный обзор (RAG-подход)...")
        
        # Используем те же параметры RAG, что и для краткой записки
        k_for_brief = {
            "market_and_finance_query": 40,
            "tech_and_assets_query": 30,
            "competitor_query": 20
        }
        main_goal_as_query = world_model.get_full_context()['static_context']['main_goal']
        relevant_kb = self._get_balanced_rag_context(world_model, main_goal_as_query, k_for_brief)
        
        lean_context = {"relevant_knowledge_base": relevant_kb}

        feedback_section = ""
        if feedback:
            feedback_section = f"""... (код feedback_section без изменений) ..."""

        prompt = f"""**ТВОЯ РОЛЬ:** ... (основной промпт без изменений) ...
**КОНТЕКСТ И БАЗА ЗНАНИЙ (ТОЛЬКО РЕЛЕВАНТНЫЕ ФАКТЫ):**
---
{json.dumps(lean_context, ensure_ascii=False, indent=2)}
---
... (остальная часть промпта) ...
{feedback_section}
Твой подробный аналитический обзор:"""
        return self._invoke_llm_for_text(prompt)
    
    def validate_artifact(self, artifact_text: str, required_sections: List[str]) -> dict:
        """
        Проверяет сгенерированный артефакт на соответствие базовым критериям качества.
        Использует быструю и дешевую модель.
        """
        print("      [Валидатор] -> Проверяю артефакт на соответствие критериям качества...")
        validator_llm = self.llms["expert_flash"]

        required_sections_str = ", ".join(required_sections)

        prompt = f"""**ТВОЯ РОЛЬ:** Ты — придирчивый ассистент-контролер качества.
**ТВОЯ ЗАДАЧА:** Проверить предоставленный текст на соответствие строгим критериям.

**КРИТЕРИИ ПРОВЕРКИ:**
1.  **Длина:** Текст должен быть длиннее 1500 символов.
2.  **Отсутствие отказов:** Текст НЕ должен содержать фразы-отказы, такие как "I cannot", "I am unable", "As an AI model", "не могу", "невозможно".
3.  **Наличие обязательных разделов:** Текст ДОЛЖЕН содержать ВСЕ следующие разделы: {required_sections_str}.

**ТЕКСТ ДЛЯ ПРОВЕРКИ:**
---
{artifact_text[:10000]}... 
---

Проанализируй текст и верни результат в формате JSON. Если все критерии выполнены, `is_valid` должно быть `True`. Если хотя бы один критерий нарушен, `is_valid` должно быть `False`, а в `reasons` укажи, что именно не так.
"""
        
        parser = PydanticOutputParser(pydantic_object=ValidationReport)
        prompt_with_instructions = f"{prompt}\n\n{parser.get_format_instructions()}"
        
        try:
            response = validator_llm.invoke(prompt_with_instructions)
            parsed_object = parser.parse(response.content)
            report = parsed_object.model_dump()
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при валидации артефакта: {e}")
            report = {"is_valid": False, "reasons": ["Ошибка парсинга ответа валидатора."]}
        # --------------------------------------------------------------------
            
        if report.get('is_valid', False):
            print("      [Валидатор] <- Артефакт прошел проверку качества.")
        else:
            reasons = report.get('reasons', ['Неизвестная ошибка валидации.'])
            print(f"      [Валидатор] <- !!! Артефакт НЕ прошел проверку. Причины: {reasons}")
            
        return report
    
    def _batch_filter_kb_by_relevance(self, knowledge_base: dict) -> dict:
        """
        Фильтрует всю базу знаний одним вызовом LLM, возвращая только коммерчески релевантные утверждения.
        Использует быструю и дешевую модель gemini-2.5-flash для экономии.
        """
        print("   [Стратег] -> Запускаю ПАКЕТНУЮ фильтрацию Базы Знаний по коммерческой релевантности...")
        
        if not knowledge_base:
            return {}

        # --- ИЗМЕНЕНИЕ: Инициализируем нужную модель прямо здесь ---
        # Это позволяет нам использовать быструю модель для этой задачи, не меняя основную модель Стратега.
        filter_llm = ChatGoogleGenerativeAI(
            model="models/gemini-2.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.0 # Нулевая температура для задач классификации
        )
        # ---------------------------------------------------------

        claims_for_analysis = []
        for claim_id, claim in knowledge_base.items():
            if claim.get('status') != 'CONFLICTED':
                claims_for_analysis.append(f"ID: {claim_id}, Утверждение: '{claim['statement']}' (Значение: {claim['value']})")
        
        claims_text = "\n".join(claims_for_analysis)

        prompt = f"""**КОНТЕКСТ:** Мы готовим бизнес-кейс для создания нового B2B HR-Tech продукта.
**ТВОЯ ЗАДАЧА:** Проанализируй список утверждений. Определи, какие из них являются **коммерчески релевантными**.

**БИЗНЕС-ВОПРОСЫ ДЛЯ ОЦЕНКИ:**
1.  Помогает ли утверждение оценить потенциальный **доход, стоимость или ROI**?
2.  Описывает ли утверждение значимый **риск** (технический, юридический, рыночный)?
3.  Раскрывает ли утверждение уникальное **конкурентное преимущество** или сильную сторону?
4.  Указывает ли утверждение на явную **потребность или "боль"** клиентов?

**СПИСОК УТВЕРЖДЕНИЙ ДЛЯ АНАЛИЗА:**
---
{claims_text}
---

**ИНСТРУКЦИЯ:** Верни JSON-объект, содержащий поле `relevant_claim_ids`. Это должен быть список, содержащий ТОЛЬКО ID тех утверждений, которые напрямую помогают ответить **хотя бы на один** из четырех бизнес-вопросов.
"""
        
        # --- ИЗМЕНЕНИЕ: Используем новый PydanticOutputParser с filter_llm ---
        parser = PydanticOutputParser(pydantic_object=BatchRelevanceReport)
        prompt_with_instructions = f"{prompt}\n\n{parser.get_format_instructions()}"
        
        try:
            response = filter_llm.invoke(prompt_with_instructions)
            parsed_object = parser.parse(response.content)
            report = parsed_object.model_dump()
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА при пакетной фильтрации: {e}")
            report = {}
        # --------------------------------------------------------------------
        
        relevant_ids = set(report.get('relevant_claim_ids', []))
        relevant_kb = {claim_id: claim for claim_id, claim in knowledge_base.items() if claim_id in relevant_ids}
        
        print(f"   [Стратег] <- Пакетная фильтрация завершена. Осталось {len(relevant_kb)} из {len(knowledge_base)} утверждений.")
        return relevant_kb
    
    def _get_balanced_rag_context(self, world_model: 'WorldModel', situation_summary: str, k_values: dict) -> dict:
        """
        Выполняет полный цикл взвешенного Multi-Query RAG и возвращает сбалансированный срез Базы Знаний.
        """
        # Шаг А: Декомпозиция мысли на запросы
        rag_queries = self._generate_rag_queries(situation_summary)
        if not rag_queries:
            print("!!! Стратег.RAG: Не удалось сгенерировать RAG-запросы.")
            return {}

        # Шаг Б: Взвешенный параллельный поиск
        print("      [Стратег.RAG] -> Выполняю взвешенный поиск по нескольким векторам...")
        all_relevant_ids = set()
        for query_name, query_text in rag_queries.items():
            if query_text:
                k = k_values.get(query_name, 10) # Используем 'k' из словаря или 10 по умолчанию
                ids = world_model.semantic_index.find_similar_claim_ids(query_text, top_k=k)
                print(f"         - Запрос '{query_name}' (k={k}): найдено {len(ids)} ID.")
                all_relevant_ids.update(ids)

        # Шаг В: Сборка и возврат обогащенного контекста
        print(f"      [Стратег.RAG] -> Всего найдено {len(all_relevant_ids)} уникальных релевантных утверждений.")
        
        full_kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        relevant_knowledge_base = {
            claim_id: full_kb[claim_id]
            for claim_id in all_relevant_ids
            if claim_id in full_kb
        }
        return relevant_knowledge_base