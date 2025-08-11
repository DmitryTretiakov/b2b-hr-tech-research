# core/llm_client.py
import sys
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from core.budget_manager import APIBudgetManager

class LLMClient:
    """
    Централизованный клиент для взаимодействия с LLM.
    Инкапсулирует иерархию моделей и управление бюджетом.
    """
    def __init__(self, budget_manager: APIBudgetManager):
        self.budget_manager = budget_manager

        # === ИЗМЕНЕНИЕ НАЧАТО: Добавлены настройки безопасности для отключения фильтров ===
        # Определяем настройки, которые отключают все блокировки.
        # Это необходимо для проверки гипотезы о том, что фильтры безопасности
        # изменяют ответ, даже если не блокируют его полностью.
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

        # Инициализация моделей с передачей настроек безопасности
        self._models = {
            # Уровень 4
            "gemini-2.5-pro": ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3, safety_settings=safety_settings),
            # Уровень 3
            "gemini-2.5-flash": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1, safety_settings=safety_settings),
            # Уровень 2
            "gemini-2.5-flash-lite": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.1, safety_settings=safety_settings),
            "gemma-3": ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", temperature=0.0, safety_settings=safety_settings),
            # Уровень 1 (пример, можно расширить)
            "gemma-3n": ChatGoogleGenerativeAI(model="models/gemma-3-12b-it", temperature=0.0, safety_settings=safety_settings),
        }
        print("-> LLMClient инициализирован с утвержденной иерархией моделей и отключенными фильтрами безопасности.")

    def _get_model_instance(self, model_name: str) -> ChatGoogleGenerativeAI:
        instance = self._models.get(model_name)
        if not instance:
            raise ValueError(f"Модель '{model_name}' не найдена в клиенте. Проверьте конфигурацию.")
        return instance

    def invoke(self, model_name: str, prompt: str):
        """
        Выполняет вызов к указанной модели с контролем бюджета и агрессивным логированием ошибок.
        """
        if not self.budget_manager.can_i_spend(model_name):
            error_message = f"Дневной лимит для модели {model_name} исчерпан."
            print(f"   [LLMClient] !!! ОШИБКА: {error_message}")
            raise ConnectionError(error_message)

        print(f"   [LLMClient] -> Вызов модели Уровня '{self.get_level(model_name)}': {model_name}")

        instance = self._get_model_instance(model_name)

        try:
            response = instance.invoke(prompt)

            if not response or not hasattr(response, 'content') or not response.content.strip():
                print("\n" + "="*80, file=sys.stderr)
                print(f"!!! [LLMClient] КРИТИЧЕСКАЯ ОШИБКА: Получен ПУСТОЙ или НЕКОРРЕКТНЫЙ ответ от модели '{model_name}'.", file=sys.stderr)
                print(f"    Сырой ответ: {response}", file=sys.stderr)
                print("="*80 + "\n", file=sys.stderr)
                sys.stderr.flush()
                raise ValueError(f"API для модели '{model_name}' вернул пустой ответ. Проверьте права доступа и квоты в Google Cloud.")

            self.budget_manager.record_spend(model_name)
            return response

        except Exception as e:
            error_details = "Дополнительные детали не найдены."
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                error_details = f"Детали ответа API: {e.response.text}"

            print("\n" + "="*80, file=sys.stderr)
            print(f"!!! [LLMClient] КРИТИЧЕСКАЯ ОШИБКА при вызове API для модели '{model_name}'.", file=sys.stderr)
            print(f"    Тип ошибки: {type(e).__name__}", file=sys.stderr)
            print(f"    Сообщение об ошибке: {e}", file=sys.stderr)
            print(f"    Дополнительно: {error_details}", file=sys.stderr)
            print("    Трассировка стека:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            sys.stderr.flush()
            raise e

    def get_level(self, model_name: str) -> int:
        """Возвращает уровень иерархии для модели."""
        if "pro" in model_name: return 4
        if "2.5-flash" in model_name and "lite" not in model_name: return 3
        if "flash-lite" in model_name or "gemma-3" in model_name: return 2
        if "gemma-3n" in model_name: return 1
        return 0