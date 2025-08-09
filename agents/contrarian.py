# agents/contrarian.py
from agents.worker import WorkerAgent

class ContrarianAgent(WorkerAgent):
    """
    Целенаправленно ищет опровержения, критику и негативные кейсы.
    Использует Gemini 2.5 Flash.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.role_prompt = """**ТЫ "АДВОКАТ ДЬЯВОЛА"**. Твоя цель — не подтвердить, а **ОПРОВЕРГНУТЬ**.
Твои поисковые запросы и выводы должны быть сформулированы так, чтобы найти критику, негативные отзывы, провальные кейсы и альтернативные мнения.
Например, вместо "преимущества X" ищи "недостатки X", "проблемы с X", "провал проекта X"."""
        print(f"-> ContrarianAgent (на базе {self.llm.model}) готов к работе.")