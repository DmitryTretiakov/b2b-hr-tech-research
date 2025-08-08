# core/budget_manager.py
import json
import os
from datetime import date

class APIBudgetManager:
    """
    Отслеживает суточное потребление API-вызовов для моделей с лимитами.
    Сбрасывает счетчики при наступлении нового дня.
    """
    def __init__(self, output_dir: str, daily_limits: dict):
        self.filepath = os.path.join(output_dir, "api_usage_log.json")
        self.limits = daily_limits
        self.usage = self._load_usage()
        print(f"-> APIBudgetManager инициализирован. Текущее использование: {self.usage}")

    def _load_usage(self) -> dict:
        """Загружает использование за СЕГОДНЯ. Если лог устарел, сбрасывает его."""
        today_str = str(date.today())
        if not os.path.exists(self.filepath):
            return {}
        
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            # Если дата в логе не совпадает с сегодняшней, значит наступил новый день.
            # Сбрасываем счетчики.
            if data.get("date") == today_str:
                return data.get("usage", {})
            else:
                print("   [Бюджет] Обнаружен новый день. Счетчики API сброшены.")
                return {}
        except (json.JSONDecodeError, IOError):
            print("!!! [Бюджет] Ошибка чтения файла использования API. Счетчики сброшены.")
            return {}

    def _save_usage(self):
        """Сохраняет текущее использование на диск."""
        today_str = str(date.today())
        try:
            with open(self.filepath, 'w') as f:
                json.dump({"date": today_str, "usage": self.usage}, f, indent=2)
        except IOError as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось сохранить лог использования API. Ошибка: {e}")

    def can_i_spend(self, model_name: str, calls_to_make: int = 1) -> bool:
        """Проверяет, можно ли совершить вызов, не превысив лимит."""
        # Если модель не отслеживается, разрешаем всегда
        if model_name not in self.limits:
            return True
            
        current_calls = self.usage.get(model_name, 0)
        limit = self.limits.get(model_name)
        
        return (current_calls + calls_to_make) <= limit

    def record_spend(self, model_name: str, calls_made: int = 1):
        """Записывает совершенный вызов и сохраняет состояние."""
        # Не отслеживаем модели без лимитов
        if model_name not in self.limits:
            return
            
        self.usage[model_name] = self.usage.get(model_name, 0) + calls_made
        limit = self.limits.get(model_name, 'N/A')
        print(f"   [Бюджет] Затрата записана. {model_name}: {self.usage[model_name]}/{limit}")
        self._save_usage()