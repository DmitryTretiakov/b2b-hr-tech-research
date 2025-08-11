# core/llm_client.py
import re
import sys
import time
import traceback
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from core.budget_manager import APIBudgetManager
from google.api_core.exceptions import ResourceExhausted


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
            # Уровень 4 - Pro
            "gemini-2.5-pro": ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3, safety_settings=safety_settings),
            "gemini-2.5-pro-creative": ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.4, safety_settings=safety_settings),
            "gemini-2.5-pro-strict": ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.05, safety_settings=safety_settings),
            # Уровень 3 - Flash
            "gemini-2.5-flash": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1, safety_settings=safety_settings),
            # Уровень 2 - Lite / Gemma
            "gemini-2.5-flash-lite": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.1, safety_settings=safety_settings),
            "gemma-3": ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", temperature=0.0, safety_settings=safety_settings),
        }
        self.FALLBACK_HIERARCHY = {
            "gemini-2.5-pro": ["gemini-2.5-flash"],
            "gemini-2.5-pro-creative": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "gemini-2.5-pro-strict": ["gemini-2.5-pro", "gemini-2.5-flash"],
            "gemini-2.5-flash": ["gemini-2.5-flash-lite"],
            "gemma-3": ["gemini-2.5-flash-lite"],
        }
        print("-> LLMClient инициализирован с иерархией фолбэков для обработки лимитов API.")
        print("-> LLMClient инициализирован с вариантами моделей (creative/strict) и отключенными фильтрами безопасности.")

    def _get_model_instance(self, model_name: str) -> ChatGoogleGenerativeAI:
        instance = self._models.get(model_name)
        if not instance:
            raise ValueError(f"Модель '{model_name}' не найдена в клиенте. Проверьте конфигурацию.")
        return instance

    def invoke(self, model_name: str, prompt: str):
        """
        Выполняет вызов к LLM с автоматическим переключением на резервные модели
        в случае исчерпания квот.
        """
        models_to_try = [model_name] + self.FALLBACK_HIERARCHY.get(model_name, [])
        
        for i, current_model in enumerate(models_to_try):
            base_model_name = re.sub(r'-(creative|strict)$', '', current_model)
            
            if not self.budget_manager.can_i_spend(base_model_name):
                print(f"   [LLMClient] -> Лимит для '{base_model_name}' исчерпан. Пропускаю.")
                continue

            print(f"   [LLMClient] -> Попытка {i+1}: вызов модели '{current_model}'")
            instance = self._get_model_instance(current_model)

            try:
                response = instance.invoke(prompt)
                if not response or not hasattr(response, 'content') or not response.content.strip():
                    raise ValueError(f"API для модели '{current_model}' вернул пустой ответ.")
                
                self.budget_manager.record_spend(base_model_name)
                return response # Успех! Выходим из цикла.

            except ResourceExhausted as e:
                print(f"   [LLMClient] !!! Квота для '{current_model}' исчерпана. Переключаюсь на следующую модель.")
                time.sleep(1) # Небольшая пауза
                continue # Переходим к следующей модели в списке
            except Exception as e:
                print(f"   [LLMClient] !!! КРИТИЧЕСКАЯ ОШИБКА (не связана с квотой) при вызове '{current_model}': {e}")
                raise e # Пробрасываем серьезные ошибки наверх

        # Если цикл завершился, значит все модели в цепочке недоступны
        raise ResourceExhausted(f"Все модели в цепочке для '{model_name}' исчерпали свои квоты.")

    def get_level(self, model_name: str) -> int:
        """Возвращает уровень иерархии для модели."""
        if "pro" in model_name: return 4
        if "2.5-flash" in model_name and "lite" not in model_name: return 3
        if "flash-lite" in model_name or "gemma-3" in model_name: return 2
        if "gemma-3n" in model_name: return 1
        return 0