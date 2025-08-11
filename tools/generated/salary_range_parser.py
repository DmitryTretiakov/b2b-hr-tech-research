import re
from typing import Dict, Any, Optional, List, Tuple

# Разрешенные импорты, даже если они не используются напрямую в этой функции.
# Это соответствует требованию о работе в строго контролируемом окружении,
# где список разрешенных импортов фиксирован для всех инструментов.
import requests
from bs4 import BeautifulSoup


def salary_range_parser(text_content: str, role: str, level: str, location: str) -> Dict[str, Any]:
    """
    Анализирует предоставленный текстовый контент для извлечения структурированных данных о зарплате.

    Инструмент ищет в тексте числовые значения, которые могут представлять собой зарплатную вилку
    (минимальная, максимальная) или среднюю зарплату. Он также определяет валюту и тип
    зарплаты (gross/net) на основе ключевых слов в тексте.

    Для повышения надежности, инструмент пытается убедиться, что найденные данные о зарплате
    контекстуально связаны с предоставленными ролью, уровнем и локацией, проверяя их
    наличие в тексте.

    Args:
        text_content (str): Текстовый контент для анализа (например, описание вакансии,
                            статья о зарплатах).
        role (str): Целевая роль для поиска (например, 'Python-разработчик').
        level (str): Целевой уровень квалификации (например, 'Middle', 'Senior').
        location (str): Целевая локация (например, 'Томск', 'Москва').

    Returns:
        Dict[str, Any]: Словарь со структурированными данными о зарплате, включая:
            - min_salary (Optional[int]): Минимальная зарплата.
            - max_salary (Optional[int]): Максимальная зарплата.
            - avg_salary (Optional[int]): Средняя зарплата.
            - currency (Optional[str]): Валюта (например, 'RUB', 'USD', 'EUR').
            - salary_type (str): Тип зарплаты ('gross', 'net' или 'unknown').
            - role_confirmed (bool): Найдено ли упоминание роли в тексте.
            - level_confirmed (bool): Найдено ли упоминание уровня в тексте.
            - location_confirmed (bool): Найдено ли упоминание локации в тексте.

    Raises:
        Exception: Если не удалось найти никаких данных о зарплате в тексте или
                   произошла непредвиденная ошибка при обработке.
    """
    try:
        # --- Вспомогательные функции и подготовка данных ---

        def _clean_number(num_str: str) -> int:
            """Очищает строку с числом от пробелов и неразрывных пробелов."""
            cleaned_str = re.sub(r'[\s\u00A0]', '', num_str)
            return int(cleaned_str)

        lower_text = text_content.lower()
        result = {
            "min_salary": None,
            "max_salary": None,
            "avg_salary": None,
            "currency": None,
            "salary_type": "unknown",
            "role_confirmed": role.lower() in lower_text if role else False,
            "level_confirmed": level.lower() in lower_text if level else False,
            "location_confirmed": location.lower() in lower_text if location else False,
        }

        # --- Шаг 1: Определение валюты ---
        currency_map = {
            '₽': 'RUB', 'руб': 'RUB', 'рублей': 'RUB',
            '$': 'USD', 'usd': 'USD', 'долларов': 'USD',
            '€': 'EUR', 'eur': 'EUR', 'евро': 'EUR',
        }
        for symbol, code in currency_map.items():
            if symbol in lower_text:
                result['currency'] = code
                break
        
        # Если валюта не найдена, дальнейший поиск бессмысленен
        if not result['currency']:
             raise Exception("Не удалось определить валюту в предоставленном тексте.")

        # --- Шаг 2: Определение типа зарплаты (Gross/Net) ---
        if any(keyword in lower_text for keyword in ['net', 'на руки', 'чистыми', 'после вычета']):
            result['salary_type'] = 'net'
        elif any(keyword in lower_text for keyword in ['gross', 'до вычета']):
            result['salary_type'] = 'gross'

        # --- Шаг 3: Извлечение зарплаты с использованием приоритетных паттернов ---
        
        # Паттерны упорядочены от наиболее специфичных (диапазон) к наименее специфичным (одно число)
        # \d[\d\s]* - цифра, за которой могут следовать другие цифры или пробелы
        salary_patterns: List[Tuple[str, List[str]]] = [
            # 1. "от X до Y"
            (r'от\s*(\d[\d\s]*)\s*до\s*(\d[\d\s]*)', ['min', 'max']),
            # 2. "X - Y"
            (r'(\d[\d\s]*)\s*[-–—]\s*(\d[\d\s]*)', ['min', 'max']),
            # 3. "от X"
            (r'от\s*(\d[\d\s]*)', ['min']),
            # 4. "до Y"
            (r'до\s*(\d[\d\s]*)', ['max']),
            # 5. Одиночное число (рассматривается как среднее)
            (r'(\d[\d\s]{4,})', ['avg']) # Ищем числа длиннее 4 знаков, чтобы избежать случайных совпадений
        ]

        found_salary = False
        for pattern, keys in salary_patterns:
            # Добавляем к паттерну контекст валюты для повышения точности
            context_pattern = pattern + r'[^a-zа-я\d]*?(?:' + '|'.join(currency_map.keys()) + r')'
            
            matches = re.finditer(context_pattern, lower_text)
            
            for match in matches:
                try:
                    if len(keys) == 2:
                        min_val = _clean_number(match.group(1))
                        max_val = _clean_number(match.group(2))
                        result['min_salary'] = min_val
                        result['max_salary'] = max_val
                    elif keys[0] == 'min':
                        result['min_salary'] = _clean_number(match.group(1))
                    elif keys[0] == 'max':
                        result['max_salary'] = _clean_number(match.group(1))
                    elif keys[0] == 'avg':
                        # Если уже есть диапазон, не перезаписываем его одиночным числом
                        if result['min_salary'] is None and result['max_salary'] is None:
                            result['avg_salary'] = _clean_number(match.group(1))
                    
                    found_salary = True
                    break # Используем первое же найденное совпадение для данного паттерна
                except (ValueError, IndexError):
                    continue # Ошибка при парсинге числа, пробуем следующее совпадение
            
            if found_salary:
                break # Если зарплата найдена, прекращаем поиск по менее приоритетным паттернам

        # --- Шаг 4: Пост-обработка и валидация ---

        # Если найден диапазон, но нет среднего, вычисляем его
        if result['min_salary'] is not None and result['max_salary'] is not None:
            if result['avg_salary'] is None:
                result['avg_salary'] = int((result['min_salary'] + result['max_salary']) / 2)
        
        # Если найдено только среднее, оно может быть и min/max
        if result['avg_salary'] is not None and result['min_salary'] is None and result['max_salary'] is None:
             result['min_salary'] = result['max_salary'] = result['avg_salary']

        # Проверяем, что хотя бы одно значение зарплаты было найдено
        if not any([result['min_salary'], result['max_salary'], result['avg_salary']]):
            raise Exception("Не удалось извлечь числовые данные о зарплате из предоставленного текста.")

        return result

    except Exception as e:
        # Перехватываем любые исключения и оборачиваем их в стандартный формат ошибки
        # Это включает как наши собственные исключения, так и непредвиденные ошибки
        raise Exception(f"Ошибка в инструменте salary_range_parser: {e}")