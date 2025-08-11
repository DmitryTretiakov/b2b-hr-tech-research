import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, Any, List, Optional, Tuple

def web_text_entity_extractor(url: str) -> Dict[str, Any]:
    """
    Извлекает структурированную информацию о проектах с веб-страницы.

    Этот инструмент анализирует HTML-содержимое по заданному URL, извлекает
    текстовые данные и идентифицирует ключевые сущности, связанные с проектом:
    - Название проекта
    - Принадлежность к организации (ТГУ)
    - Направленность (EdTech, стартап, цифровая инициатива)
    - Текущий статус (активен, в разработке, завершен, неактивен)

    Инструмент использует комбинацию анализа HTML-тегов (h1, title) для
    определения названия и поиска по ключевым словам в тексте для
    определения остальных атрибутов.

    Args:
        url: Строка с полным URL-адресом веб-страницы для анализа.

    Returns:
        Словарь, содержащий извлеченные данные в структурированном виде.
        Пример:
        {
            "project_name": "Цифровые кафедры",
            "organization": "ТГУ",
            "project_type": "EdTech",
            "status": "активен"
        }

    Raises:
        Exception: Если происходит сетевая ошибка (недоступность URL, код ответа
                   не 2xx), ошибка парсинга или если на странице не удалось
                   найти ни одной из искомых сущностей.
    """
    try:
        # 1. Получение и базовая проверка содержимого страницы
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Вызовет исключение для кодов 4xx/5xx

        # 2. Парсинг HTML и извлечение чистого текста
        soup = BeautifulSoup(response.content, 'html.parser')

        # Удаление ненужных тегов (скрипты, стили) для чистоты текста
        for element in soup(["script", "style", "nav", "footer", "aside"]):
            element.extract()

        # Получаем весь видимый текст со страницы
        full_text = soup.get_text(separator=' ', strip=True)
        # Нормализуем текст для регистронезависимого поиска
        lower_text = full_text.lower()

        # 3. Инициализация словаря для хранения результатов
        extracted_data: Dict[str, Optional[str]] = {
            "project_name": None,
            "organization": None,
            "project_type": None,
            "status": None
        }

        # 4. Логика извлечения сущностей

        # 4.1. Извлечение названия проекта (наиболее вероятные источники)
        # Приоритет: <h1>, <title>, затем поиск по паттерну "проект «...»"
        project_name_candidates: List[str] = []
        if soup.h1 and soup.h1.string:
            project_name_candidates.append(soup.h1.string.strip())
        if soup.title and soup.title.string:
            project_name_candidates.append(soup.title.string.strip())
        
        # Поиск по паттерну в тексте
        name_match = re.search(r'(?:проект|инициатива|стартап)\s*[«"“]([^»"”]+)[»"”]', full_text, re.IGNORECASE)
        if name_match:
            project_name_candidates.append(name_match.group(1).strip())

        # Выбираем самый длинный и релевантный кандидат как наиболее вероятное название
        if project_name_candidates:
            # Простая эвристика: более длинное название часто более полное
            extracted_data["project_name"] = max(project_name_candidates, key=len)


        # 4.2. Определение принадлежности к организации (ТГУ)
        tgu_keywords = ['тгу', 'томский государственный университет', 'tomsk state university']
        if any(keyword in lower_text for keyword in tgu_keywords):
            extracted_data["organization"] = "ТГУ"

        # 4.3. Определение направленности проекта
        type_map: Dict[str, List[str]] = {
            "EdTech": ["edtech", "образовательные технологии", "цифровое обучение", "образовательная платформа"],
            "стартап": ["стартап", "startup", "инновационный проект"],
            "цифровая инициатива": ["цифровая инициатива", "инициатива по цифровизации", "цифровая трансформация"]
        }
        for project_type, keywords in type_map.items():
            if any(keyword in lower_text for keyword in keywords):
                extracted_data["project_type"] = project_type
                break # Найдено первое совпадение

        # 4.4. Определение статуса проекта
        status_map: Dict[str, List[str]] = {
            "активен": ["активен", "в работе", "действующий", "поддерживается", "реализуется"],
            "в разработке": ["в разработке", "разрабатывается", "пилотный запуск", "создается", "планируется"],
            "завершен": ["завершен", "закончен", "внедрен", "реализован", "архивный"],
            "неактивен": ["неактивен", "заморожен", "приостановлен"]
        }
        for status, keywords in status_map.items():
            if any(keyword in lower_text for keyword in keywords):
                extracted_data["status"] = status
                break # Найдено первое совпадение

        # 5. Финальная проверка и возврат результата
        # Если не удалось извлечь ни одного значения, считаем задачу проваленной.
        if all(value is None for value in extracted_data.values()):
            raise Exception(f"Не удалось извлечь ни одной значимой сущности с URL: {url}. "
                            "Возможно, страница не содержит информации о проектах в явном виде.")

        return extracted_data

    except requests.exceptions.RequestException as e:
        raise Exception(f"Сетевая ошибка при доступе к URL '{url}': {e}")
    except Exception as e:
        # Перехват всех остальных исключений (включая bs4, re и наши собственные)
        raise Exception(f"Произошла внутренняя ошибка при обработке URL '{url}': {e}")