# agents/search_agent.py
import os
import json
from utils.helpers import google_search

class SearchAgent:
    """
    Отвечает за выполнение поисковых запросов и управление кэшем.
    Не содержит логики LLM, является инструментом для других агентов.
    """
    def __init__(self, api_key: str, cx_id: str, cache_dir: str):
        self.api_key = api_key
        self.cx_id = cx_id
        self.cache_path = os.path.join(cache_dir, "raw_search_cache.json")
        self.cache = self._load_cache()
        print("-> SearchAgent инициализирован и готов к работе.")

    def _load_cache(self) -> dict:
        """Загружает кэш из файла при инициализации."""
        print(f"   [Кэш] Загрузка кэша из файла: {self.cache_path}")
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache_data = json.load(f)
                print(f"   [Кэш] Успешно загружено {len(cache_data)} записей.")
                return cache_data
            except (json.JSONDecodeError, FileNotFoundError):
                print("   [Кэш] !!! Внимание: Файл кэша поврежден или пуст. Создается новый кэш.")
                return {}
        print("   [Кэш] Файл кэша не найден. Будет создан новый.")
        return {}

    def _save_cache(self):
        """Сохраняет текущее состояние кэша в файл."""
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"!!! ОШИБКА: Не удалось сохранить файл кэша. Ошибка: {e}")

    def search(self, query: str) -> dict:
        """
        Основной метод. Проверяет кэш, и если запроса там нет,
        выполняет поиск через Google API и обновляет кэш.
        """
        # Нормализуем запрос для консистентности ключей в кэше
        normalized_query = query.strip().lower()

        if normalized_query in self.cache:
            print(f"   [Кэш] Результат для '{query}' найден в кэше.")
            return self.cache[normalized_query]
        
        # Если в кэше нет, выполняем реальный поиск
        results = google_search(normalized_query, self.api_key, self.cx_id)
        
        # Если поиск прошел без ошибок, добавляем в кэш и сохраняем
        if "error" not in results:
            self.cache[normalized_query] = results
            self._save_cache()
            print(f"   [Кэш] Новый результат для '{query}' сохранен в кэш.")
        
        return results