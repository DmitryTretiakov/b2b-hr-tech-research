# tools/web_search.py
import os
import json
import requests

def perform_search(query: str, num_results: int = 10) -> dict:
    """
    Выполняет поиск с автоматическим переключением на резервный API.
    Основной: Google Custom Search. Резервный: SERPER.
    """
    # --- Попытка №1: Основной API (Google Custom Search) ---
    google_api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    google_cx_id = os.getenv("SEARCH_ENGINE_ID")
    if google_api_key and google_cx_id:
        print(f"   [WebSearchTool] -> Попытка поиска через Google API: '{query}'")
        url = "https://www.googleapis.com/customsearch/v1"
        params = {'key': google_api_key, 'cx': google_cx_id, 'q': query, 'num': num_results}
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            print("   [WebSearchTool] <- Успешный ответ от Google API.")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"   [WebSearchTool] !!! Ошибка Google API: {e}. Переключаюсь на резервный API.")

    # --- Попытка №2: Резервный API (SERPER) ---
    serper_api_key = os.getenv("SERPER_API_KEY")
    if serper_api_key:
        print(f"   [WebSearchTool] -> Попытка поиска через SERPER API: '{query}'")
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": query, "num": num_results})
        headers = {'X-API-KEY': serper_api_key, 'Content-Type': 'application/json'}
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=15)
            response.raise_for_status()
            print("   [WebSearchTool] <- Успешный ответ от SERPER API.")
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"   [WebSearchTool] !!! Ошибка резервного SERPER API: {e}.")

    # --- Полный провал ---
    print("!!! [WebSearchTool] КРИТИЧЕСКАЯ ОШИБКА: Оба поисковых API недоступны.")
    raise ConnectionError("Оба поисковых API недоступны.")