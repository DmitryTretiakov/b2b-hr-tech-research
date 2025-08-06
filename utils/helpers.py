# utils/helpers.py
import os
import requests
import re
import json

def google_search(query: str, api_key: str, cx_id: str, num_results: int = 10) -> dict:
    """
    Выполняет поиск через Google Custom Search API.
    (ИЗМЕНЕНИЕ: num_results по умолчанию увеличен до 10 для большей глубины)
    Возвращает сырой JSON ответа или словарь с ключом 'error'.
    """
    if not api_key or not cx_id:
        print("!!! ОШИБКА: Ключи для Google Search не предоставлены в .env")
        return {"error": "API-ключи не предоставлены."}
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cx_id, 'q': query, 'num': num_results}
    
    print(f"    -> [Поиск] Выполняю запрос: '{query}'...")
    try:
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status() # Проверка на ошибки HTTP (4xx, 5xx)
        print(f"    <- [Поиск] Ответ получен.")
        return response.json()
    except requests.exceptions.Timeout:
        print(f"!!! СЕТЕВАЯ ОШИБКА: Таймаут (20с) при поиске по запросу '{query}'")
        return {"error": "Таймаут запроса."}
    except requests.exceptions.RequestException as e:
        print(f"!!! СЕТЕВАЯ ОШИБКА: {e}")
        return {"error": str(e)}

def parse_json_from_response(text: str) -> any:
    """
    Надежно извлекает и парсит JSON из текстового ответа LLM.
    Справляется с markdown-блоками, окружающим текстом и ошибками.
    """
    if not isinstance(text, str):
        print("!!! ОШИБКА ПАРСИНГА: Входные данные не являются строкой.")
        return None

    # Ищем JSON внутри markdown-блока ```json ... ```
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Если не нашли markdown, ищем первый '{' или '[' и последний '}' или ']'
        start_curly = text.find('{')
        end_curly = text.rfind('}')
        start_square = text.find('[')
        end_square = text.rfind(']')

        if start_curly != -1 and end_curly != -1:
            json_str = text[start_curly:end_curly+1]
        elif start_square != -1 and end_square != -1:
            json_str = text[start_square:end_square+1]
        else:
            json_str = text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"!!! ОШИБКА ПАРСИНГА JSON: Не удалось извлечь валидный JSON.")
        return None

def sanitize_filename(text: str) -> str:
    """Очищает текст для создания безопасного имени файла с транслитерацией."""
    text = text.lower()
    translit_map = {
        'а':'a','б':'b','в':'v','г':'g','д':'d','е':'e','ё':'yo',
        'ж':'zh','з':'z','и':'i','й':'y','к':'k','л':'l','м':'m','н':'n',
        'о':'o','п':'p','р':'r','с':'s','т':'t','у':'u','ф':'f','х':'h',
        'ц':'ts','ч':'ch','ш':'sh','щ':'sch','ъ':'','ы':'y','ь':'','э':'e',
        'ю':'yu','я':'ya'
    }
    for cyr, lat in translit_map.items():
        text = text.replace(cyr, lat)
    
    text = re.sub(r'[^a-z0-9\s-]', '', text)
    text = re.sub(r'[\s-]+', '_', text).strip('_')
    return text[:70]

def format_search_results_for_llm(search_results: dict) -> str:
    """
    Форматирует сырой JSON от Google Search в удобную для LLM строку.
    """
    if "error" in search_results or not search_results.get("items"):
        return "Поиск не дал результатов или произошла ошибка."
    
    snippets = []
    for i, item in enumerate(search_results.get("items", [])):
        snippet = (
            f"--- Результат Поиска #{i+1} ---\n"
            f"Источник: {item.get('link', 'N/A')}\n"
            f"Заголовок: {item.get('title', 'N/A')}\n"
            f"Фрагмент: {item.get('snippet', 'N/A')}\n"
        )
        snippets.append(snippet)
    
    return "\n".join(snippets)