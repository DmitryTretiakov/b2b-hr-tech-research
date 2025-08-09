# agents/researcher.py
from agents.worker import WorkerAgent

class ResearcherAgent(WorkerAgent):
    """
    Ищет подтверждающие, основные факты по теме.
    Использует Gemini 2.5 Flash.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role_prompt = "Твоя цель — найти подтверждающие, основные факты, успешные кейсы и позитивные данные по теме."
        print(f"-> ResearcherAgent (на базе {self.llm.model}) готов к работе.")