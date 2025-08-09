# agents/analyst.py
import json
from typing import TYPE_CHECKING
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import AnalystReport, FullSynthesisReport

if TYPE_CHECKING:
    from core.world_model import WorldModel

class AnalystAgent:
    """
    Анализирует Базу Знаний через RAG-шлюз для извлечения инсайтов.
    Не работает со всей базой, а только с релевантным срезом.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> AnalystAgent (на базе {self.llm.model}) с RAG-шлюзом готов к работе.")

    def _get_rag_context(self, world_model: 'WorldModel', goal: str, top_k: int) -> dict:
        """
        RAG-шлюз: извлекает релевантный срез из Базы Знаний.
        """
        full_kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        if not full_kb:
            return {}
        
        print(f"   [AnalystAgent.RAG] -> Выполняю поиск {top_k} релевантных фактов для цели: '{goal}'")
        relevant_ids = world_model.semantic_index.find_similar_claim_ids(goal, top_k=top_k)
        rag_context = {claim_id: full_kb[claim_id] for claim_id in relevant_ids if claim_id in full_kb}
        
        print(f"   [AnalystAgent.RAG] <- Шлюз отобрал {len(rag_context)} из {len(full_kb)} фактов.")
        return rag_context

    def run_reflection_analysis(self, world_model: 'WorldModel', analysis_goal: str) -> dict:
        """Выполняет быстрый анализ для рефлексии, используя RAG-контекст."""
        rag_context = self._get_rag_context(world_model, analysis_goal, top_k=50)
        if not rag_context:
            return {"key_insights": ["База знаний пуста."], "data_gaps": ["Все данные отсутствуют."], "confidence_level": 0.0}

        print("   [AnalystAgent] -> Провожу анализ для рефлексии...")
        prompt = f"""**ТВОЯ РОЛЬ:** Старший системный аналитик.
**ТВОЯ ЗАДАЧА:** Проанализировать предоставленный РЕЛЕВАНТНЫЙ СРЕЗ из Базы Знаний и предоставить краткую сводку для планировщика.

**РЕЛЕВАНТНЫЙ СРЕЗ ИЗ БАЗЫ ЗНАНИЙ:**
---
{json.dumps(rag_context, ensure_ascii=False, indent=2)}
---
**ИНСТРУКЦИИ:**
1.  Найди 3-5 ключевых инсайтов.
2.  Найди 2-3 главных пробела в данных.
3.  Оцени общую уверенность в полноте данных.
4.  Верни результат в виде строгого JSON.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm, sanitizer_llm=self.sanitizer_llm,
            prompt=prompt, pydantic_schema=AnalystReport, budget_manager=self.budget_manager
        )
        print("   [AnalystAgent] <- Анализ для рефлексии завершен.")
        return report

    def run_final_synthesis(self, world_model: 'WorldModel', analysis_goal: str) -> dict:
        """Выполняет глубокий синтез, используя RAG-контекст."""
        rag_context = self._get_rag_context(world_model, analysis_goal, top_k=150)
        if not rag_context:
            return {}
            
        print("   [AnalystAgent] -> Провожу финальный синтез...")
        prompt = f"""**ТВОЯ РОЛЬ:** Главный Продуктовый Стратег, готовящий данные для писателя.
**ТВОЯ ЗАДАЧА:** Преобразовать РЕЛЕВАНТНЫЙ СРЕЗ из Базы Знаний в структурированный JSON для финального отчета.

**РЕЛЕВАНТНЫЙ СРЕЗ ИЗ БАЗЫ ЗНАНИЙ:**
---
{json.dumps(rag_context, ensure_ascii=False, indent=2)}
---
**ИНСТРУКЦИИ:**
1.  Для КАЖДОГО раздела в JSON-схеме подбери самые релевантные факты из предоставленного среза.
2.  Сформулируй каждый факт как краткий, емкий инсайт (`text`).
3.  Для каждого инсайта ОБЯЗАТЕЛЬНО укажи `source_claim_id`.
4.  Верни результат в виде ОДНОГО JSON-объекта.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm, sanitizer_llm=self.sanitizer_llm,
            prompt=prompt, pydantic_schema=FullSynthesisReport, budget_manager=self.budget_manager
        )
        print("   [AnalystAgent] <- Финальный синтез завершен.")
        return report