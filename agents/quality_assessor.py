# agents/quality_assessor.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import BatchQualityAssessmentReport, ClaimList, BaseModel
from typing import List

# --- Модель для SanityCheckCritic ---
class SanityCheckReport(BaseModel):
    """Отчет от SanityCheckCritic с перечнем утверждений, прошедших финальную проверку."""
    verified_claim_ids: List[str] = "Список ID всех утверждений, которые прошли финальную проверку на здравый смысл и коммерческую релевантнсть."

# --- Класс 1: Дешевый массовый оценщик ---
class BatchQualityAssessor:
    """
    Оценивает качество пакета утверждений по базовым критериям.
    Использует самую дешевую модель (Gemini 2.5 Flash-lite).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> BatchQualityAssessor (на базе {self.llm.model}) готов к работе.")

    def assess_batch(self, claims_batch: list[dict]) -> dict:
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
                claim['feedback'] = assessment['reason']
                results['fixable_claims'].append(claim)
            else:
                results['bad_claims'].append(claim)

        print(f"   [QualityAssessor] <- Аудит завершен. Качественных: {len(results['good_claims'])}, Требуют исправления: {len(results['fixable_claims'])}, Брак: {len(results['bad_claims'])}")
        return results

# --- Класс 2: Дорогой эксперт по здравому смыслу ---
class SanityCheckCritic:
    """
    Проводит финальную, глубокую проверку на здравый смысл и релевантность.
    Использует модель среднего класса (Gemini 2.5 Flash).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> SanityCheckCritic (на базе {self.llm.model}) готов к работе.")

    def verify_batch(self, claims_batch: list[dict]) -> list[dict]:
        """Принимает пакет 'хороших' утверждений и отсеивает сомнительные."""
        if not claims_batch:
            return []

        print(f"   [SanityCheckCritic] -> Провожу финальную проверку {len(claims_batch)} утверждений на здравый смысл...")
        prompt = f"""**ТВОЯ РОЛЬ:** Ты — самый строгий и опытный аналитик в команде. Твоя задача — быть последним рубежом обороны перед тем, как факт попадет в Базу Знаний.
**ТВОЯ ЗАДАЧА:** Тебе предоставлен пакет утверждений, которые младший аналитик (на базе Flash-lite) счел качественными. Проверь их по более строгим критериям.

**КРИТЕРИИ "ЗОЛОТОГО СТАНДАРТА":**
1.  **Коммерческая Релевантность:** Помогает ли этот факт напрямую принять инвестиционное решение? (Оценить рынок, понять риски, увидеть преимущество).
2.  **Отсутствие "Воды":** Не является ли утверждение банальной истиной или общеизвестным фактом, не несущим ценности? (Пример "воды": "Python - популярный язык программирования").
3.  **Логическая Непротиворечивость:** Нет ли в утверждении внутренних противоречий или абсурдных выводов?

**ПАКЕТ УТВЕРЖДЕНИЙ ДЛЯ ФИНАЛЬНОЙ ПРОВЕРКИ:**
---
{json.dumps(claims_batch, ensure_ascii=False, indent=2)}
---
**ИНСТРУКЦИИ:**
Верни JSON-объект, содержащий поле `verified_claim_ids`. Это должен быть список, содержащий ТОЛЬКО ID тех утверждений, которые прошли твою строгую проверку.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=SanityCheckReport,
            budget_manager=self.budget_manager
        )

        verified_ids = set(report.get('verified_claim_ids', []))
        final_claims = [claim for claim in claims_batch if claim['claim_id'] in verified_ids]

        print(f"   [SanityCheckCritic] <- Проверка завершена. {len(final_claims)} из {len(claims_batch)} утверждений прошли 'Золотой Стандарт'.")
        return final_claims