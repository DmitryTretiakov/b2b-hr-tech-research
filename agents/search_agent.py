# agents/search_agent.py
import os
import json
from utils.helpers import robust_hybrid_search

class SearchAgent:
    """
    Гибридный поисковый агент с поддержкой Google и Serper, механизмом фолбэка
    и интеллектуальным дросселированием для снижения нагрузки на API.
    """
    def __init__(self, google_api_key, google_cx_id, serper_api_key, cache_dir, primary_api='serper'):
        self.google_api_key = google_api_key
        self.google_cx_id = google_cx_id
        self.serper_api_key = serper_api_key
        self.primary_api = primary_api
        self.cache_path = os.path.join(cache_dir, "raw_search_cache.json")
        self.cache = self._load_cache()
        print(f"-> SearchAgent (Гибридный) инициализирован. Основной API: {self.primary_api.upper()}")

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
        выполняет гибридный поиск и обновляет кэш.
        """
        normalized_query = query.strip().lower()
        if normalized_query in self.cache:
            print(f"   [Кэш] Результат для '{query}' найден в кэше.")
            return self.cache[normalized_query]
        
        results = robust_hybrid_search(normalized_query)
        
        if "error" not in results:
            self.cache[normalized_query] = results
            self._save_cache()
            print(f"   [Кэш] Новый результат для '{query}' сохранен в кэш.")
        
        return results