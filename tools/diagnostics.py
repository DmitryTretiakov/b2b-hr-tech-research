# tools/diagnostics.py
import os
import json
import requests
from typing import Dict

def probe_google_api(model_name: str, api_key: str) -> Dict:
    """
    Выполняет низкоуровневый диагностический запрос к Google Generative AI API.
    Не использует langchain, чтобы получить сырую, нефильтрованную информацию.

    Args:
        model_name: Имя модели для проверки (например, 'gemini-2.5-pro').
        api_key: Ваш Google API ключ.

    Returns:
        Словарь с полной диагностической информацией о запросе.
    """
    # Формируем URL эндпоинта для REST API
    # Примечание: REST API может использовать немного другие имена моделей, чем SDK.
    # Например, 'models/gemini-2.5-pro' в SDK может соответствовать 'gemini-2.5-pro' здесь.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # Минимально необходимая структура запроса
    payload = {
        "contents": [{
            "parts": [{"text": "Это диагностический тест API. Просто ответь 'OK'."}]
        }]
    }
    
    headers = {
        "Content-Type": "application/json"
    }

    print(f"   [API Probe] -> Отправляю прямой запрос на: {url}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        
        # Пытаемся распарсить тело ответа как JSON, если не получается - возвращаем как текст
        try:
            response_body = response.json()
        except json.JSONDecodeError:
            response_body = response.text

        return {
            "status": "SUCCESS",
            "http_status_code": response.status_code,
            "response_headers": dict(response.headers),
            "response_body": response_body,
            "error_message": None
        }
    except requests.exceptions.RequestException as e:
        print(f"   [API Probe] !!! СЕТЕВАЯ ОШИБКА: {e}")
        return {
            "status": "NETWORK_ERROR",
            "http_status_code": None,
            "response_headers": None,
            "response_body": None,
            "error_message": str(e)
        }