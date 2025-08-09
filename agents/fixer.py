# agents/fixer.py
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from core.budget_manager import APIBudgetManager
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import ClaimList

class BatchFixerAgent:
    """
    Пытается исправить утверждения из "серой зоны".
    Использует модель среднего класса (Gemini 2.5 Flash).
    """
    def __init__(self, llm: ChatGoogleGenerativeAI, sanitizer_llm: ChatGoogleGenerativeAI, budget_manager: APIBudgetManager):
        self.llm = llm
        self.sanitizer_llm = sanitizer_llm
        self.budget_manager = budget_manager
        print(f"-> BatchFixerAgent (на базе {self.llm.model}) готов к работе.")

    def fix_batch(self, claims_to_fix: list[dict]) -> list[dict]:
        """Принимает пакет утверждений с полем 'feedback' и пытается их исправить."""
        if not claims_to_fix:
            return []

        print(f"   [FixerAgent] -> Пытаюсь исправить пакет из {len(claims_to_fix)} утверждений...")
        prompt = f"""**ТВОЯ РОЛЬ:** Ты — опытный редактор на фабрике данных.
**ТВОЯ ЗАДАЧА:** Тебе предоставлен пакет некачественных утверждений. Для каждого из них указана причина брака (`feedback`). Твоя задача — исправить их, сохранив исходный `claim_id`.

**ПРАВИЛА ИСПРАВЛЕНИЯ:**
1.  Внимательно прочитай `feedback`.
2.  Переформулируй `statement` и `value`, чтобы устранить недостаток.
3.  Если исправить невозможно, сохрани утверждение без изменений, но поставь `confidence_score` на 0.1.
4.  Не меняй `claim_id`.

**УТВЕРЖДЕНИЯ ДЛЯ ИСПРАВЛЕНИЯ:**
---
{json.dumps(claims_to_fix, ensure_ascii=False, indent=2)}
---
Верни ИСПРАВЛЕННЫЙ список утверждений в формате JSON.
"""
        report = invoke_llm_for_json_with_retry(
            main_llm=self.llm,
            sanitizer_llm=self.sanitizer_llm,
            prompt=prompt,
            pydantic_schema=ClaimList,
            budget_manager=self.budget_manager
        )

        fixed_claims = report.get('claims', [])
        print(f"   [FixerAgent] <- Попытка исправления завершена. Получено {len(fixed_claims)} утверждений.")
        return fixed_claims