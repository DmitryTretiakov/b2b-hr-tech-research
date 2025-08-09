# agents/chief_strategist.py
import json
import os
from pydantic import BaseModel, Field 
from typing import List, Literal, TYPE_CHECKING 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from utils.helpers import invoke_llm_for_json_with_retry
from core.budget_manager import APIBudgetManager
from google.api_core.exceptions import ResourceExhausted


# --- РЕШЕНИЕ ПРОБЛЕМЫ ЦИКЛИЧЕСКОЙ ЗАВИСИМОСТИ ТИПОВ ---
if TYPE_CHECKING:
    from core.world_model import WorldModel

# --- ОПРЕДЕЛЕНИЕ СТРУКТУРЫ ПЛАНА С ПОМОЩЬЮ PYDANTIC ---

class Task(BaseModel):
    """Описывает одну конкретную задачу в рамках фазы проекта."""
    task_id: str = Field(description="Уникальный идентификатор задачи, например 'task_001'. Должен быть уникальным во всем плане.")
    assignee: Literal['HR_Expert', 'Finance_Expert', 'Competitor_Expert', 'Tech_Expert', 'ProductOwnerAgent', 'FinancialModelAgent', 'ProductManagerAgent', 'System_Diagnostician', 'Contrarian_Expert'] = Field(description="Эксперт, которому поручена задача.")
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

class GapAnalysisReport(BaseModel):
    """Отчет по анализу пробелов в данных."""
    missing_info_tasks: List[Task] = Field(description="Список исследовательских задач для сбора недостающей критической информации. Если информация полна, список должен быть пустым.")

class ChiefStrategist:
    """
    "Мозг" системы. Создает план, проводит рефлексию и пишет финальные отчеты.
    Работает с самой мощной моделью (gemini-pro).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, medium_llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.medium_llm = medium_llm  # <--- ДОБАВЛЕНО
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print("-> ChiefStrategist (на базе gemini-pro) готов к работе. Оснащен 'санитарной' моделью и средней моделью.")

    
        
    

    # --- НОВЫЙ ПРИВАТНЫЙ МЕТОД ДЛЯ ФИЛЬТРАЦИИ БАЗЫ ЗНАНИЙ ---
    def _filter_kb_by_relevance(self, knowledge_base: dict) -> dict:
        """
        Фильтрует всю базу знаний, оставляя только коммерчески релевантные утверждения.
        Теперь использует пакетную обработку.
        """
        return self._batch_filter_kb_by_relevance(knowledge_base)

    def _invoke_llm_for_text(self, prompt: str) -> str:
      """Простой вызов LLM для генерации текста с контролем бюджета."""
      model_name = self.llm.model
      if not self.budget_manager.can_i_spend(model_name):
          print(f"!!! [Бюджет] ДНЕВНОЙ ЛИМИТ для стратегической модели {model_name} ИСЧЕРПАН.")
          raise ResourceExhausted(f"Daily budget limit reached for strategic model {model_name}")

      print(f"   [Стратег] -> Вызов LLM ({model_name}) для генерации текста...")
      try:
          response = self.llm.invoke(prompt)
          self.budget_manager.record_spend(model_name) # Записываем успешную трату
          return response.content
      except Exception as e:
          # Если это ошибка квоты, она уже была потрачена
          if isinstance(e, ResourceExhausted):
              self.budget_manager.record_spend(model_name)
          print(f"!!! КРИТИЧЕСКАЯ ОШИБКА LLM: Вызов API провалился. Ошибка: {e}")
          raise e # Перевыбрасываем ошибку, чтобы ее поймал оркестратор

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
**КЛЮЧЕВОЙ ПРИНЦИП ПЛАНИРОВАНИЯ (ИНВЕРСИЯ):** Твоя главная цель — **попытаться доказать, что предложенная идея нежизнеспособна**. Ты должен создать план, который в первую очередь ищет риски, опровержения и причины, по которым проект НЕ СЛЕДУЕТ запускать. Только если после всех попыток найти опровержение гипотеза устоит, ее можно считать коммерчески интересной.
**ТВОЯ ЗАДАЧА:** Проанализируй "ВХОДНОЙ БРИФ ДЛЯ ВЕРИФИКАЦИИ". Сгенерируй долгосрочный стратегический план из 3-4 логических фаз. 
**МАНДАТ НА РЫНОЧНУЮ ВАЛИДАЦИЮ:**

Твой план ОБЯЗАН содержать **сбалансированные** задачи, направленные на поиск как подтверждений, так и опровержений. Например:
**ЗАДАЧА А:** "Найти 3-5 прямых конкурентов 'Карьерного Навигатора', проанализировать их функционал и цены."
**ЗАДАЧА Б:** "Найти 3-5 компаний, которые пытались запустить похожий продукт и провалились. Проанализировать причины провала."
1.  **Анализ Конкурентов:** Найти 3-5 прямых конкурентов "Карьерного Навигатора", проанализировать их функционал, цены и целевую аудиторию.
2.  **Размер Рынка (TAM/SAM/SOM):** Найти отчеты и статьи с количественными оценками объема рынка корпоративного обучения и HR-Tech в России.
3.  **Подтверждение Проблемы:** Найти интервью, статьи или исследования, в которых HR-директора крупных компаний говорят о проблемах управления карьерой и развитии талантов.
**ПЕРВАЯ ФАЗА** должна быть посвящена **"Глубокой Разведке Активов ТГУ"**. Последующие фазы должны быть направлены на анализ рынка, конкурентов, разработку бизнес-кейса и MVP. Для каждой фазы создай список из 2-4 первоочередных, практически-ориентированных задач.
**ДОСТУПНЫЕ ЭКСПЕРТЫ И ИХ ВОЗМОЖНОСТИ:**
- `HR_Expert`, `Finance_Expert`, `Competitor_Expert`, `Tech_Expert`: Проводят исследования и собирают факты.
- `ProductOwnerAgent`: **СТРОГО ДЛЯ РАЗРЕШЕНИЯ КОНФЛИКТОВ.** Используй этого агента только для задач, где нужно сравнить два противоречащих факта. Не назначай ему творческие или синтезирующие задачи.
- `FinancialModelAgent`: Анализирует собранные финансовые данные и **генерирует смету и прогноз прибыли в виде таблицы**.
- `ProductManagerAgent`: Анализирует собранные требования и **генерирует детальное описание продукта и User Stories**.


Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
"""
        plan = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=StrategicPlan,
            budget_manager=self.budget_manager
        )

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
        
        reflection_context = world_model.get_reflection_context()

        # Шаг А: Осмысление ситуации (без изменений)
        situation_summary = self._summarize_situation(reflection_context)
        if not situation_summary:
            print("!!! Стратег: Не удалось осмыслить ситуацию. План не будет обновлен.")
            return reflection_context['dynamic_knowledge']['strategic_plan']

        # Шаг Б: Получение сбалансированного контекста с помощью единого RAG-метода
        # Для рефлексии нам не нужен очень широкий контекст.
        k_for_reflection = {
            "market_and_finance_query": 15,
            "tech_and_assets_query": 10,
            "competitor_query": 10
        }
        relevant_kb = self._get_balanced_rag_context(world_model, situation_summary, k_for_reflection)

        # Шаг В: Генерация обновленного плана
        updated_plan = self._generate_updated_plan(situation_summary, reflection_context, relevant_kb)

        if updated_plan and "phases" in updated_plan:
            print("   [Стратег] Рефлексия завершена. План обновлен.")
            return updated_plan
        else:
            print("!!! Стратег: Не удалось сгенерировать обновленный план. План не будет обновлен.")
            return reflection_context['dynamic_knowledge']['strategic_plan']
        
    def _generate_rag_queries(self, situation_summary: str) -> dict:
        """
        Декомпозирует общую мысль на несколько сфокусированных запросов для Multi-Query RAG.
        """
        print("      [Стратег.RAG] -> Декомпозирую общую мысль на сфокусированные запросы...")
        prompt = f"""**ТВОЯ ЗАДАЧА:** Ты — системный аналитик. Твоя цель — преобразовать общую аналитическую сводку в несколько конкретных, сжатых поисковых запросов для векторной базы данных.

**АНАЛИТИЧЕСКАЯ СВОДКА ОТ СТРАТЕГА:**
---
{situation_summary}
---

**ИНСТРУКЦИЯ:**
На основе этой сводки, сформулируй три очень коротких и сфокусированных запроса, по одному на каждый домен. Запросы должны отражать суть сводки в контексте каждого домена.

Верни результат в виде ОДНОГО JSON-объекта.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.sanitizer_llm,          # Используем модель из конструктора
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=RagQuerySet,
            budget_manager=self.budget_manager
        )
        return report
    
    def _summarize_situation(self, reflection_context: dict) -> str:
        """Шаг А рефлексии: Анализ и выводы в свободной форме."""
        print("      [Стратег.Рефлексия] Шаг А: Анализирую текущую ситуацию...")
        prompt = f"""**ТВОЯ РОЛЬ:** Главный Продуктовый Стратег.
**ТВОЯ ЗАДАЧА:** Проанализировать все имеющиеся данные по завершенной фазе исследования и написать краткую сводку (summary) в свободной форме.

**ОБЛЕГЧЕННЫЙ КОНТЕКСТ И РЕЗУЛЬТАТЫ ДЛЯ РЕФЛЕКСИИ:**
---
{json.dumps(reflection_context, ensure_ascii=False, indent=2)}

---

**ТВОЙ МЫСЛИТЕЛЬНЫЙ ПРОЦЕСС:**
1.  **Проанализируй `last_phase_summary`:** Что говорят итоги последней фазы?
1.5.  **Проанализируй проваленные задачи (status: FAILED):** В чем причина? Критична ли потеря этой информации?
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
2.  **КРИТИЧЕСКИ ВАЖНО:** После завершения текущей фазы, найди следующую фазу со статусом 'PENDING' и измени ее статус на 'IN_PROGRESS'. Это гарантирует непрерывность работы.
3.  Если Стратег решил, что нужно дополнительное исследование, добавь новые задачи в **новую активную** фазу.
4.  Если следующих фаз нет и все цели достигнуты, измени `main_goal_status` на `READY_FOR_FINAL_BRIEF`.
5.  Верни **полностью обновленный объект стратегического плана**.

Ты ОБЯЗАН вернуть результат в формате JSON, соответствующем предоставленной схеме.
"""
        return invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=StrategicPlan,
            budget_manager=self.budget_manager
        )

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

        MIN_FACTS_FOR_SUMMARY = 10 # Минимальное количество фактов для генерации
        if not relevant_kb or len(relevant_kb) < MIN_FACTS_FOR_SUMMARY:
            error_message = f"# ГЕНЕРАЦИЯ ПРОВАЛЕНА\n\nПричина: Недостаточно релевантных фактов в Базе Знаний для написания качественного отчета (найдено {len(relevant_kb)}, требуется минимум {MIN_FACTS_FOR_SUMMARY})."
            print(f"!!! [Стратег] ОШИБКА: {error_message}")
            return error_message
    
        # Создаем облегченный контекст для промпта
        lean_context = {"relevant_knowledge_base": relevant_kb}

        feedback_section = ""
        if feedback:
            feedback_section = f"""... (код feedback_section без изменений) ..."""

        prompt = f"""**ТВОЯ РОЛЬ:** Ты - Главный Продуктовый Стратег. Твоя аудитория - коммерческий директор. Его волнуют цифры, риски, ROI и конкурентные преимущества. Избегай академического языка.
**ТВОЯ ЗАДАЧА:** На основе предоставленных РЕЛЕВАНТНЫХ ФАКТОВ, напиши убедительную аналитическую записку. Объем: не более 3 страниц. Стиль: Максимально сжатый, по делу.

**КОНТЕКСТ И БАЗА ЗНАНИЙ (ТОЛЬКО РЕЛЕВАНТНЫЕ ФАКТЫ):**
---
{json.dumps(lean_context, ensure_ascii=False, indent=2)}
---

**СТРУКТУРА АНАЛИТИЧЕСКОЙ ЗАПИСКИ (2-3 СТРАНИЦЫ):**
1.  **Резюме для Руководителя (Executive Summary):** Краткая суть (возможность, решение, выгода, запрос).
2.  **Анализ Рынка и Ключевая Возможность:** Самые важные цифры и выводы.
3.  **Концепция Продукта "Карьерный Навигатор":** Краткое описание УТП.
4.  **Наше Уникальное Преимущество (Почему ТГУ?):** 2-3 главных аргумента.
5.  **Предварительная Бизнес-Модель и Риски:** Только ключевые моменты.
6.  **Дорожная Карта MVP и Следующие Шаги:** Четкий план действий и запрос.
**ПРАВИЛО ЦИТИРОВАНИЯ:** Каждое ключевое утверждение (цифры, факты о конкурентах) ОБЯЗАТЕЛЬНО должно сопровождаться ссылкой на доказательство в формате [Утверждение: claim_id].
{feedback_section}
Твоя финальная аналитическая записка:"""
        return self._invoke_llm_for_text(prompt)

    def write_extended_brief(self, world_model: 'WorldModel', feedback: str = None) -> str:
        """Пишет подробный обзор, используя RAG-контекст."""
        print("   [Стратег] Финальный Шаг (2/2): Пишу подробный обзор (RAG-подход)...")
        
        k_for_brief = {
            "market_and_finance_query": 40,
            "tech_and_assets_query": 30,
            "competitor_query": 20
        }
        main_goal_as_query = world_model.get_full_context()['static_context']['main_goal']
        relevant_kb = self._get_balanced_rag_context(world_model, main_goal_as_query, k_for_brief)

        MIN_FACTS_FOR_SUMMARY = 20 # Минимальное количество фактов для генерации
        if not relevant_kb or len(relevant_kb) < MIN_FACTS_FOR_SUMMARY:
            error_message = f"# ГЕНЕРАЦИЯ ПРОВАЛЕНА\n\nПричина: Недостаточно релевантных фактов в Базе Знаний для написания качественного отчета (найдено {len(relevant_kb)}, требуется минимум {MIN_FACTS_FOR_SUMMARY})."
            print(f"!!! [Стратег] ОШИБКА: {error_message}")
            return error_message
        
        lean_context = {"relevant_knowledge_base": relevant_kb}

        feedback_section = ""
        if feedback:
            feedback_section = f"""
**ОБРАТНАЯ СВЯЗЬ ОТ ВАЛИДАТОРА:**
Твоя предыдущая попытка была отклонена по следующим причинам:
{feedback}
**ТВОЯ ЗАДАЧА:** Перепиши подробный обзор, полностью устранив указанные недостатки.
"""
        # --- ПОЛНЫЙ, КОРРЕКТНЫЙ ПРОМПТ ---
        prompt = f"""**ТВОЯ РОЛЬ:** Ты - Главный Продуктовый Стратег. Твоя аудитория - технически подкованный Владелец Продукта, которому нужна максимальная детализация для дальнейшей работы.
**ТВОЯ ЗАДАЧА:** На основе предоставленных РЕЛЕВАНТНЫХ ФАКТОВ, напиши подробный аналитический обзор. Объем: 5-10 страниц.

**КОНТЕКСТ И БАЗА ЗНАНИЙ (ТОЛЬКО РЕЛЕВАНТНЫЕ ФАКТЫ):**
---
{json.dumps(lean_context, ensure_ascii=False, indent=2)}
---

**СТРУКТУРА ОБЗОРА (5-10 СТРАНИЦ):**
1.  **Резюме и Ключевые Стратегические Выводы.**
2.  **Глава 1: Глубокий Анализ Активов ТГУ.** Подробный разбор сильных и слабых сторон.
3.  **Глава 2: Анализ Рынка и Трендов.** Детальный обзор с цифрами и прогнозами.
4.  **Глава 3: Конкурентный Ландшафт.** Подробные досье на каждого конкурента.
5.  **Глава 4: Продуктовая Концепция.** Детальное описание "Карьерного Навигатора".
6.  **Глава 5: Бизнес-Кейс.** Включи сюда **таблицы с предварительными расчетами** сметы на MVP и финансовой модели.
7.  **Глава 6: Дорожная Карта и Техническое Задание для MVP.**
8.  **Приложение: Полный список верифицированных 'Утверждений' (Claims)** с источниками.
**ПРАВИЛО ЦИТИРОВАНИЯ:** Используй ссылки на доказательства [Утверждение: claim_id] по всему тексту.
{feedback_section}
Твой подробный аналитический обзор:"""
        return self._invoke_llm_for_text(prompt)
    
    def validate_artifact(self, validator_llm: ChatGoogleGenerativeAI, artifact_text: str, required_sections: List[str]) -> dict:
        """
        Проверяет сгенерированный артефакт на соответствие базовым критериям качества.
        Использует быструю и дешевую модель.
        """
        print("      [Валидатор] -> Проверяю артефакт на соответствие критериям качества...")

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
        
        report = invoke_llm_for_json_with_retry(
            main_llm=validator_llm,          # Основная модель - та, что передали для валидации
            sanitizer_llm=self.sanitizer_llm, # Резервная модель - из self.sanitizer_llm
            prompt=prompt,
            pydantic_schema=ValidationReport,
            budget_manager=self.budget_manager
        )
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
            
        if not report: # Если invoke_llm_for_json_with_retry вернул пустой словарь
            report = {"is_valid": False, "reasons": ["Критическая ошибка: не удалось получить ответ от валидатора."]}

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

        report = invoke_llm_for_json_with_retry(
            main_llm=self.medium_llm,             # Используем среднюю модель
            sanitizer_llm=self.sanitizer_llm, # Резервная модель - из self.sanitizer_llm
            prompt=prompt,
            pydantic_schema=BatchRelevanceReport,
            budget_manager=self.budget_manager
        )
        # --- КОНЕЦ ИЗМЕНЕНИЯ ---
        
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
    
    def generate_knowledge_base_report(self, world_model: 'WorldModel') -> str:
        """
        Создает полный, аудируемый отчет по всей верифицированной Базе Знаний.
        Это детерминированный метод, он не использует LLM.
        """
        print("   [Стратег] -> Генерирую финальный отчет по Базе Знаний...")
        kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        
        report_lines = ["# Полный Аудиторский Отчет по Базе Знаний\n\n"]
        report_lines.append("Этот документ содержит все верифицированные утверждения (факты), на которых основаны финальные аналитические отчеты.\n\n---\n\n")

        # Сортируем факты для логичной структуры, например, по ID
        sorted_claim_ids = sorted(kb.keys())

        for claim_id in sorted_claim_ids:
            claim = kb[claim_id]
            # Включаем в отчет только верифицированные и не устаревшие факты
            if claim.get('status') in ['VERIFIED', 'UNVERIFIED']:
                report_lines.append(f"## Утверждение: `{claim_id}`\n")
                report_lines.append(f"- **Формулировка:** {claim.get('statement')}\n")
                report_lines.append(f"- **Значение:** {claim.get('value')}\n")
                report_lines.append(f"- **Источник:** <{claim.get('source_link')}>\n")
                report_lines.append(f"- **Тип источника:** {claim.get('source_type')} (Доверие: {claim.get('source_trust') * 100}%)\n")
                report_lines.append(f"- **Уверенность в факте:** {claim.get('confidence_score') * 100:.1f}%\n")
                report_lines.append(f"- **Прямая цитата:**\n```\n{claim.get('source_quote')}\n```\n")
                report_lines.append("---\n\n")

        print("   [Стратег] <- Отчет по Базе Знаний сгенерирован.")
        return "".join(report_lines)
    
    def run_gap_analysis(self, world_model: 'WorldModel') -> list:
        """
        Использует мощную модель (Gemini 2.5 Flash) для анализа полноты всех собранных данных.
        Если находит пробелы, генерирует задачи для их устранения.
        """
        print("   [Стратег.GapAnalysis] -> Анализирую Базу Знаний на предмет пробелов...")
        
        full_context = world_model.get_full_context()
        kb = full_context['dynamic_knowledge']['knowledge_base']
        main_goal = full_context['static_context']['main_goal']
        
        prompt = f"""**ТВОЯ РОЛЬ:** Ты — придирчивый и дотошный Старший Аналитик, почти параноик. Твоя задача — найти слабые места в исследовании, прежде чем оно попадет к руководству.
**КОНТЕКСТ:** Команда младших аналитиков (на базе Gemma) собрала Базу Знаний для достижения главной цели: "{main_goal}". Они считают, что работа закончена. Твоя задача — доказать, что они неправы.
**ВСЯ СОБРАННАЯ БАЗА ЗНАНИЙ:**
---
{json.dumps(kb, ensure_ascii=False, indent=2)}
---
**ТВОЯ ЗАДАЧА:**
1.  **Проанализируй** всю Базу Знаний в контексте главной цели.
2.  **Найди "белые пятна".** Какой критически важной информации НЕ ХВАТАЕТ для создания убедительного бизнес-кейса, финансовой модели и продуктового брифа? Думай о рисках, скрытых расходах, юридических аспектах, реалистичных цифрах по рынку РФ/Томска, технических ограничениях.
3.  **Сгенерируй список задач.** Если ты нашел пробелы, сформулируй от 1 до 5 четких исследовательских задач для младших аналитиков (assignee: 'HR_Expert', 'Finance_Expert', 'Tech_Expert' и т.д.).
4.  **Если данные полны,** верни пустой список `missing_info_tasks`. Это твой способ сказать "Проверено, данных достаточно".

Ты ОБЯЗАН вернуть результат в формате JSON.
"""
        # Используем self.medium_llm (Gemini 2.5 Flash) для этой ответственной задачи
        report = invoke_llm_for_json_with_retry(
            main_llm=self.medium_llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=GapAnalysisReport,
            budget_manager=self.budget_manager
        )
        
        tasks = report.get('missing_info_tasks', [])
        if tasks:
            print(f"   [Стратег.GapAnalysis] <- Найдены пробелы в данных. Сгенерировано {len(tasks)} задач.")
        else:
            print("   [Стратег.GapAnalysis] <- Пробелов не найдено. Данные признаны полными.")
            
        return tasks
