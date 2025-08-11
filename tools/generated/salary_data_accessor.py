import os
import json
import requests
from bs4 import BeautifulSoup

def salary_data_accessor(query: str) -> dict:
    """
    Извлекает предварительно собранные данные о зарплатах из локального файла.

    Этот инструмент предназначен для работы в контролируемой среде, где предыдущая задача
    ('task_c01_salary_collection') уже собрала данные и сохранила их в стандартизированном
    местоположении. Инструмент ищет файл с именем 'task_c01_salary_collection.json'
    в текущем рабочем каталоге.

    Файл должен иметь формат JSON и представлять собой словарь (объект), где ключами
    являются поисковые запросы (например, названия должностей), а значениями -
    собранные по ним данные.

    Args:
        query (str): Ключ для поиска в файле данных. Обычно это название
                     должности или другая сущность, по которой были собраны
                     данные о зарплате. Запрос не чувствителен к регистру.

    Returns:
        dict: Словарь, содержащий найденные данные о зарплате для указанного
              запроса. Структура возвращаемого словаря зависит от данных,
              собранных на предыдущем шаге.

    Raises:
        Exception: Если файл 'task_c01_salary_collection.json' не найден.
        Exception: Если файл имеет неверный формат JSON.
        Exception: Если указанный 'query' (ключ) не найден в файле данных.
        Exception: При возникновении других непредвиденных ошибок во время
                   чтения файла или обработки данных.
    """
    file_name = "task_c01_salary_collection.json"
    
    try:
        # Шаг 1: Проверить существование файла с данными
        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Файл с данными '{file_name}' не найден. Убедитесь, что задача 'task_c01_salary_collection' была выполнена успешно.")

        # Шаг 2: Прочитать и распарсить JSON файл
        with open(file_name, 'r', encoding='utf-8') as f:
            try:
                data_collection = json.load(f)
            except json.JSONDecodeError as e:
                raise Exception(f"Ошибка декодирования JSON в файле '{file_name}': {e}")

        if not isinstance(data_collection, dict):
            raise Exception(f"Ожидалось, что корневой элемент в '{file_name}' будет словарем (объектом JSON), но получен {type(data_collection).__name__}.")

        # Шаг 3: Найти данные по ключу (запросу)
        # Приводим ключи в данных и сам запрос к нижнему регистру для нечувствительного к регистру поиска
        normalized_query = query.lower()
        
        # Создаем словарь с ключами в нижнем регистре для поиска
        normalized_data = {k.lower(): v for k, v in data_collection.items()}

        if normalized_query in normalized_data:
            found_data = normalized_data[normalized_query]
            return {
                "status": "success",
                "query": query,
                "data": found_data
            }
        else:
            raise KeyError(f"Данные для запроса '{query}' не найдены в файле '{file_name}'.")

    except FileNotFoundError as e:
        # Эта ошибка уже содержит подробное сообщение
        raise Exception(str(e))
    except KeyError as e:
        # Эта ошибка также содержит подробное сообщение
        raise Exception(str(e))
    except Exception as e:
        # Перехват всех остальных непредвиденных ошибок
        raise Exception(f"Произошла непредвиденная ошибка при доступе к данным о зарплате: {e}")