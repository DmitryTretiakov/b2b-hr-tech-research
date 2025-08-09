# agents/quality_assessor.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import BatchQualityAssessmentReport, Claim

class BatchQualityAssessor:
    """
    Оценивает качество пакета утверждений.
    Использует самую дешевую модель (Gemini 2.5 Flash-lite).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> BatchQualityAssessor (на базе {self.llm.model}) готов к работе.")

    def assess_batch(self, claims_batch: list[dict]) -> dict:
        """Оценивает пакет утверждений и разделяет их на три группы."""
        if not claims_batch:
            return {'good_claims': [], 'fixable_claims': [], 'bad_claims': []}

        print(f"   [QualityAssessor] -> Провожу аудит пакета из {len(claims_batch)} утверждений...")
        prompt = f"""**ТВОЯ РОЛЬ:** Ты — придирчивый контролер качества на фабрике данных.
**ТВОЯ ЗАДАЧA:** Для КАЖДОГО утверждения в пакете, вынеси вердикт по чек-листу.

**ЧЕК-ЛИСТ ПРОВЕРКИ:**
1.  **Конкретность:** Утверждение содержит конкретные цифры, названия, даты? (Плохо: "высокая зарплата", хорошо: "зарплата 150 тыс. руб.")
2.  **Доказуемость:** `source_quote` действительно подтверждает `statement`?
3.  **Полнота:** Вся ли ключевая информация на месте? (например, для зарплаты - грейд, город).

**ПАКЕТ УТВЕРЖДЕНИЙ ДЛЯ АУДИТА:**
---
{json.dumps(claims_batch, ensure_ascii=False, indent=2)}
---
**ИНСТРУКЦИИ:**
Верни JSON-отчет. Для каждого утверждения укажи:
- `is_ok`: `true`, если все пункты чек-листа пройдены.
- `is_fixable`: `true`, если есть мелкие недостатки (например, не хватает города), но в целом факт ценный.
- `reason`: причина, если `is_ok` равно `false`.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=BatchQualityAssessmentReport,
            budget_manager=self.budget_manager
        )

        # Разделяем утверждения на группы
        results = {'good_claims': [], 'fixable_claims': [], 'bad_claims': []}
        assessments = {assessment['claim_id']: assessment for assessment in report.get('assessments', [])}

        for claim in claims_batch:
            assessment = assessments.get(claim['claim_id'])
            if not assessment:
                results['bad_claims'].append(claim)
                continue

            if assessment['is_ok']:
                results['good_claims'].append(claim)
            elif assessment['is_fixable']:
                # Добавляем причину для исправления
                claim['feedback'] = assessment['reason']
                results['fixable_claims'].append(claim)
            else:
                results['bad_claims'].append(claim)

        print(f"   [QualityAssessor] <- Аудит завершен. Качественных: {len(results['good_claims'])}, Требуют исправления: {len(results['fixable_claims'])}, Брак: {len(results['bad_claims'])}")
        return results