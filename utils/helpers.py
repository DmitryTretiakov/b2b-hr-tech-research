# utils/helpers.py
import os
import requests
import re
import json
import time
from serpapi import SerpApiClient
import time
from langchain.output_parsers import PydanticOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel
from google.api_core.exceptions import ResourceExhausted
from core.budget_manager import APIBudgetManager

# --- НОВОЕ КАСТОМНОЕ ИСКЛЮЧЕНИЕ ---
class SearchAPIFailureError(Exception):
    """Исключение, выбрасываемое, когда все поисковые API не смогли вернуть результат."""
    pass

def robust_hybrid_search(query: str, num_results: int = 10) -> dict:
    """
    Выполняет поиск, используя Serper (через requests) как основной API и Google как резервный.
    """
    # --- Попытка №1: Основной API (Serper) с использованием requests ---
    serper_api_key = os.getenv("SERPER_API_KEY") # Убедитесь, что имя в .env совпадает!

    # --- ДИАГНОСТИКА: ПРОВЕРКА КЛЮЧА ВНУТРИ ФУНКЦИИ ---
    print(f"\n--- ДИАГНОСТИКА ВНУТРИ robust_hybrid_search ---")
    print(f"   Запрос: '{query}'")
    if serper_api_key:
        print(f"   [OK] Ключ SERPER_API_KEY доступен внутри функции.")
    else:
        print("   [!!! ОШИБКА] Ключ SERPER_API_KEY НЕ доступен внутри функции (равен None).")
    print("------------------------------------------\n")
    # --- КОНЕЦ ДИАГНОСТИКИ ---

    if serper_api_key:
        print(f"    -> [Поиск Serper] Выполняю POST-запрос: '{query}'...")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": num_results, "gl": "ru", "hl": "ru"})
        headers = {
            'X-API-KEY': serper_api_key,
            'Content-Type': 'application/json'
        }
        
        retries = 3
        delay = 2
        for i in range(retries):
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=30)
                response.raise_for_status() # Проверка на ошибки HTTP (4xx, 5xx)
                
                results = response.json()
                
                if "organic" in results:
                    # Адаптируем ответ Serper под наш стандартный формат
                    formatted_results = {
                        "items": [{"title": item.get("title"), "link": item.get("link"), "snippet": item.get("snippet")}
                                  for item in results["organic"][:num_results]]
                    }
                    print(f"    <- [Поиск Serper] Ответ получен и обработан.")
                    return formatted_results
                else:
                    print(f"   [Поиск Serper] Запрос выполнен, но органические результаты не найдены.")
                    break # Прерываем попытки, переходим к резервному API

            except requests.exceptions.HTTPError as e:
                # Особая обработка ошибки "Too Many Requests"
                if e.response.status_code == 429:
                    print(f"   [Поиск Serper] !!! Внимание: Получен статус 429. Попытка {i+1}/{retries}. Жду {delay} сек...")
                    time.sleep(delay)
                    delay *= 2
                    continue
                else:
                    print(f"!!! СЕТЕВАЯ ОШИБКА HTTP (Serper): {e}")
                    break # Прерываем попытки при других ошибках
            
            except Exception as e:
                print(f"   [Поиск Serper] !!! ОШИБКА: {e}. Попытка {i+1}/{retries}. Жду {delay} сек...")
                time.sleep(delay)
                delay *= 2
        
        print(f"!!! [SearchAgent] Основной API (Serper) не справился после {retries} попыток.")

    # --- Попытка №2: Резервный API (Google Custom Search) ---
    # Этот блок остается без изменений
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cx_id = os.getenv("SEARCH_ENGINE_ID")
    if google_api_key and google_cx_id:
        print(f"!!! [SearchAgent] Переключаюсь на резервный API (Google)...")
        # google_search_legacy выбросит SearchAPIFailureError в случае провала
        return google_search_legacy(query, google_api_key, google_cx_id, num_results)

    # Если мы дошли до сюда, значит, оба API провалились.
    print("!!! КРИТИЧЕСКАЯ ОШИБКА ПОИСКА: Все API недоступны или не справились.")
    raise SearchAPIFailureError(f"Все поисковые API провалились для запроса: '{query}'")

def google_search_legacy(query: str, api_key: str, cx_id: str, num_results: int) -> dict:
    """Резервная функция поиска через Google API с механизмом повторных попыток."""
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cx_id, 'q': query, 'num': num_results}
    retries = 3
    delay = 2
    
    print(f"    -> [Поиск Google] Выполняю запрос: '{query}'...")
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            print(f"    <- [Поиск Google] Ответ получен.")
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"   [Поиск Google] !!! Внимание: Получен статус 429 (Too Many Requests). Попытка {i+1}/{retries}. Жду {delay} сек...")
                time.sleep(delay)
                delay *= 2
                continue
            else:
                print(f"!!! СЕТЕВАЯ ОШИБКА HTTP (Google): {e}")
                return {"error": str(e)}
        except Exception as e:
            print(f"!!! СЕТЕВАЯ ОШИБКА (Google): {e}")
            time.sleep(delay)
            delay *= 2
            continue
            
    # --- ИЗМЕНЕНИЕ: ВМЕСТО ВОЗВРАТА СЛОВАРЯ, ВЫБРАСЫВАЕМ ИСКЛЮЧЕНИЕ ---
    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА ПОИСКА (Google): Все {retries} попытки провалились.")
    raise SearchAPIFailureError(f"Резервный API (Google) не справился после {retries} попыток.")

def google_search(query: str, api_key: str, cx_id: str, num_results: int = 10) -> dict:
    """
    Выполняет поиск через Google Custom Search API с механизмом повторных попыток.
    Возвращает сырой JSON ответа или словарь с ключом 'error'.
    """
    if not api_key or not cx_id:
        print("!!! ОШИБКА: Ключи для Google Search не предоставлены в .env")
        return {"error": "API-ключи не предоставлены."}
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {'key': api_key, 'cx': cx_id, 'q': query, 'num': num_results}
    
    retries = 3
    delay = 2 # Начальная задержка в секундах
    
    print(f"    -> [Поиск] Выполняю запрос: '{query}'...")
    for i in range(retries):
        try:
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status() # Проверка на ошибки HTTP (4xx, 5xx)
            print(f"    <- [Поиск] Ответ получен.")
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            # Особая обработка ошибки "Too Many Requests"
            if e.response.status_code == 429:
                print(f"   [Поиск] !!! Внимание: Получен статус 429 (Too Many Requests). Попытка {i+1}/{retries}. Жду {delay} сек...")
                time.sleep(delay)
                delay *= 2 # Увеличиваем задержку для следующей попытки
                continue
            else:
                print(f"!!! СЕТЕВАЯ ОШИБКА HTTP: {e}")
                return {"error": str(e)}
        
        except requests.exceptions.Timeout:
            print(f"!!! СЕТЕВАЯ ОШИБКА: Таймаут (20с) при поиске по запросу '{query}'. Попытка {i+1}/{retries}.")
            time.sleep(delay)
            delay *= 2
            continue
            
        except requests.exceptions.RequestException as e:
            print(f"!!! СЕТЕВАЯ ОШИБКА: {e}")
            return {"error": str(e)}

    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА ПОИСКА: Все {retries} попытки для запроса '{query}' провалились.")
    return {"error": f"Все {retries} попытки провалились."}


def robust_json_parser(text: str) -> any:
    """
    Улучшенная версия парсера. Ищет JSON более агрессивно.
    """
    if not isinstance(text, str):
        print("!!! ОШИБКА ПАРСИНГА: Входные данные не являются строкой.")
        return None

    # 1. Попытка найти JSON в markdown-блоке
    match = re.search(r'```json\s*([\s\S]*?)\s*```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("!!! ОШИБКА ПАРСИНГА: Найден markdown-блок, но внутри невалидный JSON.")
            # Продолжаем, чтобы попробовать другие методы

    # 2. Попытка найти первый '{' или '[' и последний '}' или ']'
    start = -1
    end = -1
    
    start_curly = text.find('{')
    end_curly = text.rfind('}')
    start_square = text.find('[')
    end_square = text.rfind(']')

    if start_curly != -1 and end_curly != -1:
        start, end = start_curly, end_curly + 1
    elif start_square != -1 and end_square != -1:
        start, end = start_square, end_square + 1
    
    if start != -1:
        json_str = text[start:end]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print(f"!!! ОШИБКА ПАРСИНГА: Не удалось распарсить фрагмент.")
            # Продолжаем, чтобы попробовать последнюю попытку

    # 3. Последняя попытка: просто пытаемся распарсить весь текст
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА ПАРСИНГА: Не удалось извлечь валидный JSON из ответа.")
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
    Форматирует сырой JSON от Google Search и Serper в удобную для LLM строку.
    """
    # Serper использует ключ 'organic', Google - 'items'
    items = search_results.get("organic", search_results.get("items", []))
    
    if "error" in search_results or not items:
        return "Поиск не дал результатов или произошла ошибка."
    
    snippets = []
    for i, item in enumerate(items):
        snippet = (f"--- Результат Поиска #{i+1} ---\n"
                   f"Источник: {item.get('link', 'N/A')}\n"
                   f"Заголовок: {item.get('title', 'N/A')}\n"
                   f"Фрагмент: {item.get('snippet', 'N/A')}\n")
        snippets.append(snippet)
    
    return "\n".join(snippets)

def invoke_llm_for_json_with_retry(
    main_llm: ChatGoogleGenerativeAI,
    sanitizer_llm: ChatGoogleGenerativeAI,
    prompt: str,
    pydantic_schema: BaseModel,
    budget_manager: APIBudgetManager,
    max_retries: int = 3
) -> dict:
    """
    Выполняет вызов LLM для получения JSON с многоуровневой стратегией повторных попыток.
    Принцип Нулевого Доверия: мы не верим, что LLM вернет валидный JSON с первого раза.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_schema)
    prompt_with_instructions = f"{prompt}\n\n{parser.get_format_instructions()}"
    
    last_error = None
    raw_output = ""

    for attempt in range(max_retries):
        print(f"      [JSON Invoker] Попытка {attempt + 1}/{max_retries}...")
        
        current_prompt = prompt_with_instructions
        current_llm = main_llm

        model_name = current_llm.model
        if not budget_manager.can_i_spend(model_name):
            print(f"!!! [Бюджет] ДНЕВНОЙ ЛИМИТ для {model_name} исчерпан. Попытка отменена.")
            last_error = ResourceExhausted(f"Daily budget limit for {model_name} reached.")
            continue # Переходим к следующей попытке (которая может использовать другую модель)

        if attempt == 1: # Вторая попытка: просим основную модель исправить свой же вывод
            print("      [JSON Invoker] Стратегия 2: Прошу основную модель исправить свой невалидный JSON.")
            current_prompt = f"""Твой предыдущий ответ не удалось распарсить. Он вернул ошибку: {last_error}.
Пожалуйста, верни ТОЛЬКО валидный JSON, который соответствует запрошенной схеме.

Вот твой предыдущий, невалидный ответ:
---
{raw_output}
---

Вот оригинальные инструкции по формату:
{parser.get_format_instructions()}
"""
        elif attempt == 2: # Третья попытка: используем "санитарную" модель для извлечения JSON
            print("      [JSON Invoker] Стратегия 3: Использую 'санитарную' модель для извлечения JSON из вывода.")
            current_llm = sanitizer_llm
            current_prompt = f"""Извлеки валидный JSON объект из текста ниже. Верни ТОЛЬКО сам JSON и ничего больше.

ТЕКСТ ДЛЯ АНАЛИЗА:
---
{raw_output}
---

Вот оригинальные инструкции по формату, которым должен соответствовать JSON:
{parser.get_format_instructions()}
"""
        try:
            response = current_llm.invoke(current_prompt)
            raw_output = response.content # Сохраняем сырой вывод для возможных исправлений
            budget_manager.record_spend(model_name)
            parsed_object = parser.parse(raw_output)
            print("      [JSON Invoker] <- Ответ LLM успешно получен и распарсен.")
            return parsed_object.model_dump()
        except Exception as e:
            if isinstance(e, ResourceExhausted):
                 budget_manager.record_spend(model_name) # Записываем даже неудачную попытку, т.к. она была сделана
            last_error = e
            print(f"      [JSON Invoker] !!! Ошибка на попытке {attempt + 1}: {e}")
            time.sleep(3) # Небольшая пауза перед следующей попыткой

    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось получить валидный JSON после {max_retries} попыток.")
    if isinstance(last_error, ResourceExhausted):
        raise last_error
    return {} # Возвращаем пустой словарь в случае полного провала
