import os
import json
import requests
import re
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Optional

# --- Вспомогательные функции, инкапсулированные внутри файла ---

def _perform_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Выполняет поиск с автоматическим переключением на резервный API.
    Возвращает список словарей с ключами 'title', 'link', 'snippet'.
    """
    # Попытка №1: Google Custom Search
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cx_id = os.getenv("SEARCH_ENGINE_ID")
    if google_api_key and google_cx_id:
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {'key': google_api_key, 'cx': google_cx_id, 'q': query, 'num': num_results}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json().get("items", [])
        except requests.exceptions.RequestException:
            pass # Игнорируем ошибку и переходим к резервному варианту

    # Попытка №2: SERPER
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key:
        try:
            url = "https://google.serper.dev/search"
            payload = json.dumps({"q": query, "num": num_results})
            headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
            response = requests.post(url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            return response.json().get("organic", [])
        except requests.exceptions.RequestException:
            pass

    raise Exception("Оба поисковых API (Google, SERPER) недоступны или не настроены.")

def _read_webpage(url: str) -> str:
    """Извлекает основной текстовый контент с веб-страницы."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            element.decompose()
        return re.sub(r'\s+', ' ', soup.get_text(separator=' ', strip=True)).strip()
    except Exception:
        return "" # Возвращаем пустую строку в случае любой ошибки чтения страницы

def _extract_salary_with_llm(text: str, query: str) -> Optional[Dict[str, Any]]:
    """
    Использует LLM для извлечения данных о зарплате из текста.
    Примечание: В реальной системе здесь был бы вызов к LLMClient.
    Для демонстрации мы имитируем этот вызов и используем простое регулярное выражение как fallback.
    """
    # --- Имитация вызова LLM ---
    # В реальной системе здесь был бы код:
    # prompt = f"Проанализируй текст и найди зарплату для '{query}'. Верни JSON..."
    # response = llm_client.invoke("gemini-2.5-flash", prompt)
    # return json.loads(response.content)
    
    # --- Упрощенный Fallback на регулярных выражениях для демонстрации ---
    text = text.replace('\xa0', ' ').replace(' ', '') # Убираем неразрывные пробелы
    
    # Паттерн: [от] 100000 [до/-] 200000 [руб/₽]
    match = re.search(r"(?:от)?(\d{5,8})(?:до|-)(\d{5,8})(руб|₽)", text, re.IGNORECASE)
    if match:
        return {
            "min_salary": int(match.group(1)),
            "max_salary": int(match.group(2)),
            "currency": "RUB"
        }
    return None

def smart_salary_extractor(query: str) -> Dict[str, Any]:
    """
    Надежно извлекает информацию о диапазоне зарплат, используя многошаговую стратегию.
    1. Выполняет поиск через API, чтобы получить список релевантных страниц.
    2. Посещает каждую страницу и извлекает ее текстовое содержимое.
    3. Анализирует текст с помощью LLM (имитация) для поиска упоминаний о зарплате.
    4. Возвращает первое найденное совпадение.

    Args:
        query (str): Поисковый запрос (должность и регион).

    Returns:
        Dict[str, Any]: Словарь с данными о зарплате и источником.

    Raises:
        Exception: Если не удалось найти информацию после проверки нескольких источников.
    """
    print(f"   [SmartSalaryExtractor] -> Начинаю умный поиск зарплаты для: '{query}'")
    try:
        search_results = _perform_search(query)
        if not search_results:
            raise Exception("Поисковый API не вернул результатов.")

        for item in search_results:
            url = item.get("link")
            if not url:
                continue
            
            print(f"      [SmartSalaryExtractor] Проверяю источник: {url}")
            page_text = _read_webpage(url)
            
            if not page_text:
                continue

            salary_data = _extract_salary_with_llm(page_text, query)
            if salary_data:
                print(f"      [SmartSalaryExtractor] <- Данные успешно найдены в {url}")
                salary_data["source"] = url
                return salary_data
        
        raise Exception(f"Не удалось найти информацию о зарплате для '{query}' после проверки {len(search_results)} источников.")

    except Exception as e:
        raise Exception(f"Инструмент smart_salary_extractor завершился с ошибкой: {e}")