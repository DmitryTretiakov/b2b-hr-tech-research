# agents/report_writer.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import FinalReport, ValidationReport
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from core.world_model import WorldModel

class ReportWriterAgent:
    """
    Отвечает за написание финальных отчетов на основе JSON от Аналитика.
    Использует самую мощную модель (Gemini 2.5 Pro).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> ReportWriterAgent (на базе {self.llm.model}) готов к работе.")

    # ПРИМЕЧАНИЕ: Эти методы будут доработаны в Фазе 3 для приема JSON от Аналитика.
    # Пока мы просто переносим старую логику RAG-генерации.
    def write_final_report(self, world_model: 'WorldModel', report_type: str, feedback: str = None) -> str:
        """Пишет отчет, используя RAG-контекст."""
        print(f"   [ReportWriter] Пишу отчет типа: {report_type}...")

        # Эта логика будет заменена на прием JSON от AnalystAgent
        # Пока оставляем RAG для совместимости
        main_goal_as_query = world_model.get_full_context()['static_context']['main_goal']
        relevant_ids = world_model.semantic_index.find_similar_claim_ids(main_goal_as_query, top_k=50)
        full_kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
        relevant_kb = {claim_id: full_kb[claim_id] for claim_id in relevant_ids if claim_id in full_kb}

        if not relevant_kb:
            return "# ГЕНЕРАЦИЯ ПРОВАЛЕНА\n\nПричина: Недостаточно фактов."

        prompt = f"""**ТВОЯ РОЛЬ:** Ты - Исполнительный Писатель.
**ТВОЯ ЗАДАЧА:** Напиши отчет '{report_type}', основываясь на фактах.

**БАЗА ЗНАНИЙ:**
---
{json.dumps(relevant_kb, ensure_ascii=False, indent=2)}
---

**ПРАВИЛО ЦИТИРОВАНИЯ:** Каждое ключевое утверждение ОБЯЗАТЕЛЬНО должно сопровождаться ссылкой на доказательство в формате [Утверждение: claim_id].

Твой финальный отчет:
"""
        # Этот вызов будет заменен на invoke_llm_for_json_with_retry с моделью FinalReport
        response = self.llm.invoke(prompt)
        self.budget_manager.record_spend(self.llm.model)
        return response.content

    def validate_artifact(self, artifact_text: str, required_sections: List[str]) -> dict:
        """Проверяет сгенерированный артефакт на соответствие базовым критериям качества."""
        print("      [Валидатор] -> Проверяю артефакт...")
        # Используем "санитарную" модель для экономии
        validator_llm = self.sanitizer_llm
        required_sections_str = ", ".join(required_sections)

        prompt = f"""**ТВОЯ РОЛЬ:** Ты — придирчивый ассистент-контролер качества.
**ТВОЯ ЗАДАЧА:** Проверить предоставленный текст на соответствие строгим критериям.

**КРИТЕРИИ ПРОВЕРКИ:**
1.  **Длина:** Текст должен быть длиннее 1000 символов.
2.  **Отсутствие отказов:** Текст НЕ должен содержать фразы-отказы.
3.  **Наличие обязательных разделов:** Текст ДОЛЖЕН содержать ВСЕ следующие разделы: {required_sections_str}.

**ТЕКСТ ДЛЯ ПРОВЕРКИ:**
---
{artifact_text[:10000]}...
---

Проанализируй текст и верни результат в формате JSON.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=validator_llm,
            sanitizer_llm=validator_llm, # Используем ее же как резервную
            prompt=prompt,
            pydantic_schema=ValidationReport,
            budget_manager=self.budget_manager
        )
        return report