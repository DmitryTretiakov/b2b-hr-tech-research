import requests
from bs4 import BeautifulSoup
import re

def webpage_reader(url: str) -> str:
    """
    Извлекает и возвращает чистое текстовое содержимое с веб-страницы по заданному URL.

    Эта функция выполняет HTTP GET-запрос к указанному URL, парсит полученный HTML-контент
    с помощью BeautifulSoup, удаляет ненужные элементы (скрипты, стили, навигацию, футеры)
    и возвращает основной текст страницы в виде единой строки.

    Args:
        url (str): URL-адрес веб-страницы, с которой необходимо прочитать содержимое.

    Returns:
        str: Очищенный текстовый контент страницы.

    Raises:
        Exception: Вызывается в следующих случаях:
            - Не удалось получить доступ к URL (сетевые ошибки, ошибки DNS).
            - Сервер вернул код ошибки HTTP (например, 404 Not Found, 500 Internal Server Error).
            - Не удалось извлечь текстовое содержимое со страницы (например, страница пуста
              или содержит только нетекстовые элементы после очистки).
            - Произошла любая другая непредвиденная ошибка в процессе обработки.
    """
    try:
        # Установка заголовков, чтобы имитировать запрос от реального браузера.
        # Это помогает избежать блокировки со стороны некоторых сайтов, которые
        # отклоняют запросы от стандартного User-Agent библиотеки requests.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,ru;q=0.8',
            'Connection': 'keep-alive',
        }

        # Выполнение запроса с таймаутом, чтобы избежать бесконечного ожидания.
        # response.raise_for_status() автоматически вызовет исключение для кодов ответа 4xx/5xx.
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        # Использование response.content и 'html.parser' для надежного парсинга
        # различных кодировок.
        soup = BeautifulSoup(response.content, 'html.parser')

        # Удаление тегов, которые обычно не содержат основного контента.
        # Это наиболее надежный способ очистки, не зависящий от конкретной
        # структуры сайта (в отличие от поиска по id='main' или class='content').
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            element.decompose()

        # Извлечение текста. separator=' ' соединяет текстовые блоки пробелом,
        # а strip=True удаляет лишние пробелы в начале и конце каждого блока.
        text = soup.get_text(separator=' ', strip=True)

        # Дополнительная очистка от множественных пробелов и пустых строк.
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            raise Exception("Не удалось извлечь текстовое содержимое со страницы. Возможно, страница пуста или содержит только нетекстовые элементы.")

        return text

    except requests.exceptions.RequestException as e:
        # Обработка всех ошибок, связанных с запросом (сеть, DNS, таймаут, HTTP-статус).
        raise Exception(f"Ошибка при доступе к URL '{url}': {e}")
    except Exception as e:
        # Обработка всех остальных возможных ошибок (например, ошибок парсинга BeautifulSoup).
        raise Exception(f"Произошла непредвиденная ошибка при обработке страницы '{url}': {e}")