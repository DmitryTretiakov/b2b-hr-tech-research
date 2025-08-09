# utils/prompt_engine.py
from typing import List, Dict

def create_dynamic_prompt(base_prompt: str, task: Dict, relevant_facts: List[Dict]) -> str:
    """
    Генерирует динамический, обогащенный промпт для Worker-агентов.

    Args:
        base_prompt: Статический ролевой промпт агента (e.g., "Ты - Researcher...").
        task: Объект текущей задачи.
        relevant_facts: Список релевантных фактов из Базы Знаний.

    Returns:
        Новый, обогащенный промпт.
    """
    
    context_section = ""
    if relevant_facts:
        facts_str = "\n".join([f"- {fact['statement']} [Утверждение: {fact['claim_id']}]" for fact in relevant_facts])
        context_section = f"""
**ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ УЖЕ НАЙДЕННОГО:**
{facts_str}

**УТОЧНЕННОЕ ЗАДАНИЕ:**
Не ищи общую информацию по теме. Сфокусируйся на поиске фактов, которые дополняют или оспаривают (в зависимости от твоей роли) уже известную информацию. Найди более глубокие, конкретные и неочевидные детали.
"""

    dynamic_prompt = f"""{base_prompt}

**ТЕКУЩАЯ ЗАДАЧА:** {task.get('description', 'Нет описания')}
**ЦЕЛЬ ЗАДАЧИ:** {task.get('goal', 'Нет цели')}
{context_section}
"""
    return dynamic_prompt