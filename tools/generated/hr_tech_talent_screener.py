import requests
from bs4 import BeautifulSoup
import re
from typing import Dict, Any, List, Set

def hr_tech_talent_screener(url: str) -> Dict[str, Any]:
    """
    Анализирует содержимое веб-страницы по заданному URL для определения,
    является ли компания B2B-поставщиком в сфере HR-Tech, специализирующимся
    на решениях для управления талантами.

    Инструмент выполняет следующие шаги:
    1. Загружает HTML-содержимое страницы по URL.
    2. Извлекает весь видимый текст и заголовок страницы для анализа.
    3. Пытается определить название компании из мета-тегов или заголовка.
    4. Анализирует текст на наличие ключевых слов, связанных с:
        - HR-Tech (Human Resources Technology)
        - B2B (Business-to-Business)
        - Управлением талантами (Talent Management)
        - Построением карьерных треков (Career Pathing)
        - Внутренними рынками талантов (Internal Talent Marketplace)
    5. На основе анализа выносит вердикт о релевантности компании и
       определяет конкретные области предлагаемых решений.

    Args:
        url (str): URL-адрес веб-страницы компании для анализа.

    Returns:
        Dict[str, Any]: Словарь с результатами анализа, содержащий:
            - "company_name" (str): Извлеченное название компании.
            - "is_relevant" (bool): True, если компания соответствует критериям, иначе False.
            - "relevant_areas" (List[str]): Список найденных релевантных областей решений.

    Raises:
        Exception: В случае сетевых ошибок (недоступность URL, ошибки HTTP),
                   ошибок парсинга или если страница не содержит текста для анализа.
    """
    try:
        # 1. Загрузка страницы с имитацией браузера для обхода простых защит
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # Вызовет исключение для кодов 4xx/5xx

    except requests.exceptions.RequestException as e:
        raise Exception(f"Сетевая ошибка при доступе к URL {url}: {e}")

    try:
        # 2. Парсинг HTML и извлечение текста
        soup = BeautifulSoup(response.content, 'html.parser')

        # Удаляем скрипты и стили, чтобы они не мешали анализу
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Извлекаем весь видимый текст, приводя к нижнему регистру для удобства поиска
        page_text = soup.get_text(separator=' ', strip=True).lower()

        if not page_text:
            raise Exception("Не удалось извлечь текст со страницы. Возможно, страница пуста или состоит из JavaScript.")

        # 3. Определение названия компании
        company_name = "Unknown"
        # Приоритетный способ: мета-тег 'og:site_name'
        og_site_name_tag = soup.find('meta', property='og:site_name')
        if og_site_name_tag and og_site_name_tag.get('content'):
            company_name = og_site_name_tag.get('content').strip()
        # Запасной способ: заголовок страницы <title>
        elif soup.title and soup.title.string:
            # Очищаем заголовок от общих слов и разделителей
            title_text = soup.title.string.strip()
            company_name = re.split(r'\s*\||–|-\s*', title_text)[0].strip()

        # 4. Определение ключевых слов и категорий
        # Используем множества (set) для быстрой проверки наличия
        hr_tech_keywords: Set[str] = {
            'hr tech', 'hrtech', 'human resources', 'управление персоналом', 'hr-платформа',
            'hr-решения', 'кадровый', 'hr-автоматизация', 'hr-сервис', 'people management'
        }

        b2b_keywords: Set[str] = {
            'b2b', 'for business', 'enterprise', 'для бизнеса', 'для компаний',
            'корпоративный', 'решения для бизнеса', 'our customers', 'наши клиенты', 'request a demo', 'запросить демо'
        }

        talent_solutions_map: Dict[str, Set[str]] = {
            "Управление талантами": {
                'talent management', 'управление талантами', 'развитие талантов', 'talent development',
                'performance management', 'управление эффективностью'
            },
            "Построение карьерных треков": {
                'career pathing', 'career tracking', 'карьерный трек', 'карьерное развитие',
                'план развития', 'career framework', 'карьерные пути'
            },
            "Внутренний рынок талантов": {
                'internal mobility', 'talent marketplace', 'internal talent', 'внутренняя мобильность',
                'рынок талантов', 'внутренний рынок талантов', 'internal opportunities'
            }
        }
        
        # 5. Анализ и вынесение вердикта
        is_hr_tech = any(keyword in page_text for keyword in hr_tech_keywords)
        is_b2b = any(keyword in page_text for keyword in b2b_keywords)

        found_areas: List[str] = []
        for area_name, keywords in talent_solutions_map.items():
            if any(keyword in page_text for keyword in keywords):
                found_areas.append(area_name)

        # Критерий релевантности: компания должна быть HR-Tech, B2B и предлагать хотя бы одно из целевых решений.
        is_relevant = is_hr_tech and is_b2b and len(found_areas) > 0

        # Формирование итогового результата
        result = {
            "company_name": company_name,
            "is_relevant": is_relevant,
            "relevant_areas": sorted(found_areas) # Сортируем для консистентности вывода
        }

        return result

    except Exception as e:
        # Перехватываем любые другие неожиданные ошибки во время парсинга или анализа
        raise Exception(f"Внутренняя ошибка при анализе страницы {url}: {e}")