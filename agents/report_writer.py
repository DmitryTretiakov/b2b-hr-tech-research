# agents/report_writer.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import FinalReport

class ReportWriterAgent:
    """
    Превращает структурированный JSON от Аналитика в связный Markdown-текст.
    Не думает, а только пишет. Использует Gemini 2.5 Pro.
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> ReportWriterAgent (на базе {self.llm.model}) готов к работе.")

    def write_final_report(self, synthesis_report: dict) -> str:
        """
        Принимает структурированный JSON и генерирует из него Markdown-отчет.
        """
        print("   [ReportWriter] -> Превращаю структурированные данные в отчет...")
        prompt = f"""**ТВОЯ РОЛЬ:** Профессиональный технический писатель и копирайтер.
**ТВОЯ ЗАДАЧА:** Тебе предоставлены структурированные данные в формате JSON. Преврати их в красивый, логичный и хорошо читаемый отчет в формате Markdown.

**СТРУКТУРИРОВАННЫЕ ДАННЫЕ ОТ АНАЛИТИКА:**
---
{json.dumps(synthesis_report, ensure_ascii=False, indent=2)}
---

**ИНСТРУКЦИИ ПО НАПИСАНИЮ:**
1.  Используй `title` из каждого раздела как заголовок (`##`).
2.  Преврати каждый `insight` в полноценное предложение или абзац.
3.  **КРИТИЧЕСКИ ВАЖНО:** После каждого инсайта, основанного на факте, вставь специальный маркер для цитаты в формате `[CITE:{{source_claim_id}}]`. Замени `{{source_claim_id}}` на реальный ID из данных.
4.  Не добавляй никакой информации от себя. Твоя задача - только изложение.
5.  Верни результат как ОДИН JSON-объект с полем `markdown_content`.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=FinalReport,
            budget_manager=self.budget_manager
        )
        print("   [ReportWriter] <- Отчет сгенерирован.")
        return report.get("markdown_content", "# ОШИБКА ГЕНЕРАЦИИ ОТЧЕТА")