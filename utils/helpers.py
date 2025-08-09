# utils/helpers.py
import json
import re
import time
from collections import OrderedDict
from typing import Type, Dict
from pydantic import BaseModel, ValidationError
from langchain.output_parsers import PydanticOutputParser
from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager

def format_search_results_for_llm(search_results: dict) -> str:
    """Форматирует сырой JSON от поисковика в удобную для LLM строку."""
    items = search_results.get("organic", search_results.get("items", []))
    if not items:
        return "Поиск не дал результатов."
    
    snippets = []
    for i, item in enumerate(items):
        snippet = (f"--- Результат Поиска #{i+1} ---\n"
                   f"Источник: {item.get('link', 'N/A')}\n"
                   f"Заголовок: {item.get('title', 'N/A')}\n"
                   f"Фрагмент: {item.get('snippet', 'N/A')}\n")
        snippets.append(snippet)
    return "\n".join(snippets)

def citation_post_processor(markdown_text: str, knowledge_base: dict) -> str:
    """Находит маркеры [CITE:claim_id], заменяет их на сноски и генерирует список источников."""
    print("   [PostProcessor] -> Обрабатываю цитаты в финальном отчете...")
    found_ids = re.findall(r'\[CITE:([\w_,-]+)\]', markdown_text)
    unique_ids_in_order = list(OrderedDict.fromkeys(found_ids))
    
    if not unique_ids_in_order:
        return markdown_text

    citation_map = {claim_id: i + 1 for i, claim_id in enumerate(unique_ids_in_order)}

    def replace_marker(match):
        claim_id = match.group(1)
        return f"[^{citation_map.get(claim_id, '??')}]"

    processed_text = re.sub(r'\[CITE:([\w_,-]+)\]', replace_marker, markdown_text)

    references_list = ["\n\n---\n\n## Список Источников\n"]
    for claim_id, number in citation_map.items():
        claim_data = knowledge_base.get(claim_id)
        if claim_data:
            source_link = claim_data.get('source_link', '#')
            statement = claim_data.get('statement', 'Утверждение не найдено.')
            references_list.append(f"[^{number}]: {statement} ([Источник]({source_link}))")
    
    return processed_text + "\n".join(references_list)

def invoke_llm_for_json_with_retry(
    llm_client: LLMClient,
    model_name: str,
    sanitizer_model_name: str,
    prompt: str,
    pydantic_schema: Type[BaseModel],
    budget_manager: APIBudgetManager,
    max_retries: int = 3
) -> Dict:
    """
    Выполняет вызов LLM для получения JSON с многоуровневой стратегией самокоррекции.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_schema)
    prompt_with_instructions = f"{prompt}\n\n{parser.get_format_instructions()}"
    
    raw_output = ""
    for attempt in range(max_retries):
        print(f"      [JSON Invoker] Попытка {attempt + 1}/{max_retries}...")
        
        current_prompt = prompt_with_instructions
        current_model = model_name

        if attempt == 1: # Вторая попытка: просим ту же модель исправить себя
            print("      [JSON Invoker] Стратегия 2: Самокоррекция.")
            current_prompt = f"""Твой предыдущий ответ не удалось распарсить.
Вот твой невалидный ответ:
---
{raw_output}
---
Пожалуйста, исправь свой JSON и верни ТОЛЬКО валидный JSON, который соответствует оригинальной схеме.
Оригинальные инструкции по формату:
{parser.get_format_instructions()}
"""
        elif attempt == 2: # Третья попытка: эскалация на "санитарную" модель
            print(f"      [JSON Invoker] Стратегия 3: Эскалация на санитарную модель '{sanitizer_model_name}'.")
            current_model = sanitizer_model_name
            current_prompt = f"""Извлеки валидный JSON объект из текста ниже. Верни ТОЛЬКО сам JSON и ничего больше.
ТЕКСТ ДЛЯ АНАЛИЗА:
---
{raw_output}
---
Вот оригинальные инструкции по формату, которым должен соответствовать JSON:
{parser.get_format_instructions()}
"""
        try:
            response = llm_client.invoke(current_model, current_prompt)
            raw_output = response.content
            parsed_object = parser.parse(raw_output)
            print("      [JSON Invoker] <- Ответ LLM успешно получен и распарсен.")
            return parsed_object.model_dump()
        
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"      [JSON Invoker] !!! Ошибка валидации/парсинга на попытке {attempt + 1}: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"      [JSON Invoker] !!! Критическая ошибка API на попытке {attempt + 1}: {e}")
            # При ошибках API (например, лимиты) нет смысла продолжать
            return {}

    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить валидный JSON после {max_retries} попыток.")
    return {}