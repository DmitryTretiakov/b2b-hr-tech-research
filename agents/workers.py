# agents/workers.py
import json
from datetime import datetime, timezone
from agents.base_agent import BaseAgent
from tools.web_search import perform_search
from utils.helpers import format_search_results_for_llm, invoke_llm_for_json_with_retry
from agents.models import FactExtractionReport, BatchQualityAssessmentReport, AnalystReport, FinalReport, SanityCheckReport

class BaseResearchAgent(BaseAgent):
    """Общий базовый класс для Researcher и Contrarian."""
    role_prompt: str = "Твоя роль: Ассистент-исследователь."

    def execute(self, task: dict, model_name: str) -> list:
        print(f"   [{self.__class__.__name__}] -> Задача '{task['task_id']}' на модели {model_name}...")
        prompt_queries = f"{self.role_prompt}\nТвоя задача: '{task['description']}'. Сгенерируй 3-4 точных поисковых запроса. Верни их как JSON-список строк."
        response_queries = self.llm_client.invoke(model_name, prompt_queries)
        try:
            queries = json.loads(response_queries.content)
        except:
            queries = [task['description']]

        search_results_text = ""
        for q in queries:
            results = perform_search(q)
            search_results_text += format_search_results_for_llm(results) + "\n\n"

        prompt_claims = f"{self.role_prompt}\nПроанализируй текст и извлеки 3-5 ключевых фактов. Заполни все поля, включая `claim_id`, `statement`, `source_link` и `source_quote`.\nТЕКСТ:\n{search_results_text}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt_claims,
            FactExtractionReport, self.budget_manager
        )

        if not report or 'extracted_facts' not in report:
            return []

        for fact in report['extracted_facts']:
            fact['created_at'] = datetime.now(timezone.utc).isoformat()
        return report['extracted_facts']

class ResearcherAgent(BaseResearchAgent):
    role_prompt: str = "Твоя роль: Ассистент-исследователь. Цель — найти подтверждающие, основные факты."

class ContrarianAgent(BaseResearchAgent):
    role_prompt: str = "Твоя роль: 'Адвокат Дьявола'. Цель — найти опровержения, критику, провальные кейсы."

class QualityAssessorAgent(BaseAgent):
    """Оценивает качество фактов и возвращает отчет."""
    def execute(self, facts_to_assess: list, model_name: str) -> dict:
        print(f"   [QualityAssessorAgent] -> Оцениваю {len(facts_to_assess)} фактов на модели {model_name}...")
        prompt = f"Твоя роль: Контролер качества. Для КАЖДОГО факта вынеси вердикт по чек-листу (Конкретность, Доказуемость, Полнота) и верни JSON-отчет.\nФАКТЫ:\n{json.dumps(facts_to_assess, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash-lite", prompt,
            BatchQualityAssessmentReport, self.budget_manager
        )
        return report

class FixerAgent(BaseAgent):
    """Исправляет некачественные факты и возвращает список исправленных."""
    def execute(self, facts_to_fix: list, model_name: str) -> list:
        print(f"   [FixerAgent] -> Исправляю {len(facts_to_fix)} фактов на модели {model_name}...")
        prompt = f"Твоя роль: Редактор. Исправь факты на основе `feedback`, сохранив `claim_id`. Если исправить невозможно, не включай факт в ответ.\nФАКТЫ:\n{json.dumps(facts_to_fix, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            FactExtractionReport, self.budget_manager
        )
        return report.get('extracted_facts', [])

class SanityCheckCritic(BaseAgent):
    """Проводит финальную проверку на здравый смысл."""
    def execute(self, facts_to_check: list, model_name: str) -> list:
        print(f"   [SanityCheckCritic] -> Финальная проверка {len(facts_to_check)} фактов на модели {model_name}...")
        prompt = f"Твоя роль: Старший аналитик. Проверь факты на коммерческую релевантность и отсутствие 'воды'. Верни JSON со списком `verified_claim_ids` тех, кто прошел проверку.\nФАКТЫ:\n{json.dumps(facts_to_check, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            SanityCheckReport, self.budget_manager
        )
        verified_ids = set(report.get('verified_claim_ids', []))
        return [fact for fact in facts_to_check if fact['claim_id'] in verified_ids]

class AnalystAgent(BaseAgent):
    """Синтезирует инсайты из Базы Знаний."""
    def execute_reflection(self, knowledge_base: dict, model_name: str) -> dict:
        print(f"   [AnalystAgent] -> Синтезирую инсайты для рефлексии на модели {model_name}...")
        prompt = f"""Твоя роль: Старший аналитик. Проанализируй Базу Знаний и предоставь краткую сводку для планировщика: 3-5 ключевых инсайтов и 2-3 пробела в данных.
БАЗА ЗНАНИЙ:
---
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
---
"""
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            AnalystReport, self.budget_manager
        )
        return {"status": "SUCCESS", "data": report}

class ReportWriterAgent(BaseAgent):
    """Пишет отчет по структурированным данным."""
    def execute(self, analysis_data: dict, model_name: str) -> dict:
        print(f"   [ReportWriterAgent] -> Пишу отчет на модели {model_name}...")
        prompt = f"""Твоя роль: Профессиональный копирайтер. Преврати структурированные данные в связный Markdown-отчет. После каждого инсайта вставь маркер [CITE:{{source_claim_id}}].
ДАННЫЕ ОТ АНАЛИТИКА:
---
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
---
"""
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash-lite", prompt,
            FinalReport, self.budget_manager
        )
        return {"status": "SUCCESS", "data": report.get('markdown_content', '')}