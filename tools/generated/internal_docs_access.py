import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urlencode

def internal_docs_access(query: str) -> dict:
    """
    Осуществляет поиск по внутренним системам документации, файловым хранилищам
    и базам данных LMS IDO для извлечения релевантных отчетов и технической документации.

    Инструмент имитирует доступ к нескольким внутренним конечным точкам:
    - Основная база знаний (по умолчанию)
    - Хранилище отчетов (активируется префиксом "report:")
    - LMS IDO (активируется префиксом "lms:")

    Для результатов в формате HTML инструмент пытается извлечь заголовок и краткое
    содержание страницы для предоставления более полного контекста.

    Args:
        query (str): Поисковый запрос. Может содержать префиксы ("report:", "lms:")
                     для направления поиска в конкретную систему.

    Returns:
        dict: Словарь, содержащий статус выполнения и найденные данные.
              Пример успешного ответа:
              {
                  "status": "success",
                  "data": [
                      {
                          "title": "Название документа",
                          "source": "URL или путь к файлу",
                          "summary": "Краткое содержание или тип документа."
                      },
                      ...
                  ]
              }

    Raises:
        Exception: Вызывается при сетевых ошибках, ошибках парсинга ответа,
                   некорректном статусе ответа от API или других непредвиденных
                   проблемах.
    """
    API_ENDPOINTS = {
        "docs": "https://internal.knowledge-base.corp/api/v1/search",
        "reports": "https://internal.reports-storage.corp/api/v2/query",
        "lms": "https://internal.lms-ido.corp/api/search"
    }
    
    target_endpoint_key = "docs"
    search_query = query

    if query.lower().startswith("report:"):
        target_endpoint_key = "reports"
        search_query = query[len("report:"):].strip()
    elif query.lower().startswith("lms:"):
        target_endpoint_key = "lms"
        search_query = query[len("lms:"):].strip()

    if not search_query:
        raise Exception("Поисковый запрос не может быть пустым.")

    target_url = API_ENDPOINTS[target_endpoint_key]
    params = {"q": search_query}
    headers = {
        "User-Agent": "InternalDocsAccessTool/1.0",
        "Accept": "application/json"
    }

    try:
        # Имитация вызова внутреннего API
        # В реальной среде здесь будет происходить реальный сетевой запрос.
        # Для демонстрации мы создадим фиктивный ответ, который мог бы вернуть такой API.
        # Этот блок закомментирован, чтобы показать, как бы выглядел реальный код.
        """
        response = requests.get(
            target_url,
            params=params,
            headers=headers,
            timeout=15
        )
        response.raise_for_status()  # Вызовет исключение для кодов 4xx/5xx
        api_data = response.json()
        """

        # --- Начало блока имитации ---
        # Вместо реального запроса, мы генерируем фиктивный ответ,
        # чтобы продемонстрировать логику обработки данных.
        if target_endpoint_key == "docs":
            mock_response_data = {
                "count": 2,
                "results": [
                    {"title": "Authentication Service Deployment Guide", "url": "https://internal.knowledge-base.corp/docs/auth-v2-deployment"},
                    {"title": "Data Pipeline Monitoring", "url": "https://internal.knowledge-base.corp/docs/data-pipeline-monitoring"}
                ]
            }
        elif target_endpoint_key == "reports":
            mock_response_data = {
                "status": "ok",
                "files": [
                    {"name": "Q3_2023_Financial_Report.pdf", "path": "smb://files/reports/Q3_2023_Financial_Report.pdf", "type": "pdf"},
                    {"name": "Annual_Performance_Review_2022.docx", "path": "smb://files/reports/Annual_Performance_Review_2022.docx", "type": "document"}
                ]
            }
        else: # lms
            mock_response_data = {
                "items": [
                    {"course_name": "Advanced Python for Secure Environments", "module_url": "https://internal.lms-ido.corp/courses/SEC-PY-301/index.html"},
                    {"asset_name": "Diagram: System Architecture", "asset_url": "https://internal.lms-ido.corp/assets/sys-arch-v3.png"}
                ]
            }
        api_data = mock_response_data
        # --- Конец блока имитации ---

        if not api_data:
            raise Exception("API вернул пустой ответ.")

        processed_results = []
        
        # Обработка ответа в зависимости от источника
        results_list = api_data.get("results") or api_data.get("files") or api_data.get("items", [])

        for item in results_list:
            title = item.get("title") or item.get("name") or item.get("course_name") or item.get("asset_name", "Без названия")
            url = item.get("url") or item.get("path") or item.get("module_url") or item.get("asset_url")
            summary = f"Тип ресурса: {item.get('type', 'webpage')}"

            if url and url.startswith("http"):
                try:
                    # Имитация вторичного запроса для получения контента страницы
                    # В реальной среде здесь был бы еще один вызов requests.get(url, ...)
                    # Для демонстрации мы используем фиктивный HTML.
                    mock_html_content = f"""
                    <html>
                        <head><title>Сводка по: {title}</title></head>
                        <body>
                            <h1>{title}</h1>
                            <p>Это автоматически сгенерированное краткое содержание для документа, найденного по запросу '{search_query}'. Документ описывает ключевые аспекты и лучшие практики.</p>
                            <div>Дополнительная информация...</div>
                        </body>
                    </html>
                    """
                    # page_response = requests.get(url, timeout=10)
                    # page_response.raise_for_status()
                    # soup = BeautifulSoup(page_response.text, 'html.parser')
                    
                    soup = BeautifulSoup(mock_html_content, 'html.parser')
                    
                    if soup.title and soup.title.string:
                        title = soup.title.string
                    
                    first_p = soup.find('p')
                    if first_p and first_p.string:
                        summary = first_p.string.strip()

                except requests.exceptions.RequestException:
                    # Не удалось загрузить страницу, используем базовую информацию
                    summary = "Не удалось загрузить предпросмотр страницы."
                except Exception:
                    # Ошибка парсинга, используем базовую информацию
                    summary = "Ошибка при обработке содержимого страницы."

            processed_results.append({
                "title": title,
                "source": url,
                "summary": summary
            })

        if not processed_results:
            return {"status": "success", "data": [{"summary": "По вашему запросу ничего не найдено."}]}

        return {"status": "success", "data": processed_results}

    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка сети при доступе к внутреннему API: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        raise Exception(f"Ошибка обработки ответа от API: неверный формат данных. {e}")
    except Exception as e:
        # Перехват всех остальных исключений, включая response.raise_for_status()
        raise Exception(f"Произошла непредвиденная ошибка при выполнении запроса: {e}")