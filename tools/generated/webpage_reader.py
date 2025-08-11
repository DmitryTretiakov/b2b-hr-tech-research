import requests
from bs4 import BeautifulSoup
import re

def webpage_reader(url: str) -> str:
    """
    Извлекает основной текстовый контент с веб-страницы по заданному URL.

    Эта функция выполняет HTTP GET-запрос к указанному URL, парсит HTML-содержимое
    и пытается извлечь основной текст, удаляя навигационные элементы, скрипты,
    стили, подвалы и боковые панели.

    Args:
        url (str): URL-адрес веб-страницы, которую необходимо прочитать.

    Returns:
        str: Очищенный основной текстовый контент страницы в виде одной строки.

    Raises:
        Exception: Если происходит ошибка сети (например, таймаут, ошибка DNS),
                   если сервер возвращает код ошибки HTTP (4xx или 5xx),
                   если не удается найти контент на странице, или если возникает
                   любая другая непредвиденная ошибка в процессе выполнения.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Устанавливаем таймаут для предотвращения "зависания" инструмента
        response = requests.get(url, headers=headers, timeout=15)
        
        # Проверяем, что запрос был успешным (коды 2xx)
        response.raise_for_status()

        # Устанавливаем кодировку на основе заголовков, если возможно, иначе utf-8
        response.encoding = response.apparent_encoding or 'utf-8'

        soup = BeautifulSoup(response.text, 'html.parser')

        # Удаляем ненужные теги, которые не содержат основного контента
        for element in soup(['script', 'style', 'header', 'footer', 'nav', 'aside', 'form']):
            element.decompose()

        # Получаем текст, заменяя разделители на пробел и удаляя лишние пробелы по краям
        text = soup.get_text(separator=' ', strip=True)

        # Сжимаем множественные пробельные символы (включая переносы строк) в один пробел
        cleaned_text = re.sub(r'\s+', ' ', text).strip()

        if not cleaned_text:
            raise Exception("Не удалось извлечь текстовый контент со страницы.")

        return cleaned_text

    except requests.exceptions.RequestException as e:
        raise Exception(f"Сетевая ошибка при доступе к URL '{url}': {e}")
    except Exception as e:
        # Перехватываем все остальные возможные ошибки, включая BeautifulSoup ошибки или наши собственные
        raise Exception(f"Произошла ошибка при обработке страницы '{url}': {e}")