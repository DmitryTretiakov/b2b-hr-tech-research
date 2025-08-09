# agents/analyst.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import AnalystReport, FullSynthesisReport

class AnalystAgent:
    """
    Анализирует всю Базу Знаний для извлечения инсайтов и синтеза.
    Не пишет текст, а генерирует структурированный JSON.
    Использует самую мощную модель (Gemini 2.5 Pro).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> AnalystAgent (на базе {self.llm.model}) готов к работе.")

    def run_reflection_analysis(self, knowledge_base: dict) -> dict:
        """
        Выполняет быстрый анализ для цикла рефлексии.
        Возвращает краткий отчет о состоянии дел.
        """
        if not knowledge_base:
            return {"key_insights": ["База знаний пуста."], "data_gaps": ["Все данные отсутствуют."], "confidence_level": 0.0}

        print("   [AnalystAgent] -> Провожу анализ для рефлексии...")
        prompt = f"""**ТВОЯ РОЛЬ:** Старший системный аналитик.
**ТВОЯ ЗАДАЧА:** Проанализировать всю имеющуюся Базу Знаний и предоставить краткую сводку для планировщика.

**ВСЯ БАЗА ЗНАНИЙ:**
---
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
---

**ИНСТРУКЦИИ:**
1.  **Найди 3-5 ключевых инсайтов:** Что самое важное мы узнали?
2.  **Найди 2-3 главных пробела в данных:** Какой критически важной информации все еще не хватает?
3.  **Оцени общую уверенность:** Насколько полны данные для принятия следующего решения?
4.  Верни результат в виде строгого JSON.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=AnalystReport,
            budget_manager=self.budget_manager
        )
        print("   [AnalystAgent] <- Анализ для рефлексии завершен.")
        return report

    def run_final_synthesis(self, knowledge_base: dict) -> dict:
        """
        Выполняет глубокий синтез всей Базы Знаний для создания финального отчета.
        """
        print("   [AnalystAgent] -> Провожу финальный синтез всей Базы Знаний...")
        prompt = f"""**ТВОЯ РОЛЬ:** Главный Продуктовый Стратег, который готовит структурированные данные для писателя.
**ТВОЯ ЗАДАЧА:** Преобразовать всю Базу Знаний в структурированный JSON для финального отчета. Ты не пишешь текст, ты извлекаешь и структурируешь инсайты.

**ВСЯ БАЗА ЗНАНИЙ:**
---
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
---

**ИНСТРУКЦИИ:**
1.  Проанализируй ВСЕ утверждения.
2.  Для КАЖДОГО раздела в JSON-схеме (`executive_summary`, `market_analysis` и т.д.) подбери самые релевантные факты.
3.  Сформулируй каждый факт как краткий, емкий инсайт (`text`).
4.  Для каждого инсайта ОБЯЗАТЕЛЬНО укажи `source_claim_id` - ID самого важного факта, который его подтверждает.
5.  Верни результат в виде ОДНОГО JSON-объекта.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=FullSynthesisReport,
            budget_manager=self.budget_manager
        )
        print("   [AnalystAgent] <- Финальный синтез завершен.")
        return report