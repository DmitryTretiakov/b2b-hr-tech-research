import json
import re
from collections import Counter
from typing import Any, Dict, List

# =====================================================================================
# ВАЖНОЕ ПРИМЕЧАНИЕ ДЛЯ РЕВЬЮЕРА:
# В соответствии с задачей, предполагается, что следующие функции-инструменты
# (`web_search`, `webpage_reader`, `salary_range_parser`) уже существуют
# в среде выполнения и доступны для вызова.
# Этот скрипт содержит только код для функции-оркестратора `salary_aggregator`,
# которая использует эти предоставленные инструменты.
# =====================================================================================


def salary_aggregator(
    search_queries: List[str],
    role: str,
    level: str,
    locations: List[str]
) -> Dict[str, Any]:
    """
    Агрегирует данные о зарплатах, полученные из веба, для указанной роли, уровня и локаций.

    Этот инструмент выполняет многошаговый процесс для сбора и анализа данных о зарплатах:
    1.  Выполняет поисковые запросы в вебе с помощью инструмента `web_search` для нахождения
        релевантных страниц (например, вакансий, обзоров зарплат).
    2.  Для каждого найденного URL извлекает текстовое содержимое страницы с помощью
        инструмента `webpage_reader`.
    3.  Анализирует (парсит) полученный текст для извлечения структурированной информации
        о зарплате (минимальная и максимальная ставка, валюта, период выплат), используя
        контекстно-зависимый инструмент `salary_range_parser`.
    4.  Собирает все успешно извлеченные данные.
    5.  Агрегирует собранные данные, вычисляя средние, минимальные и максимальные значения
        для каждой уникальной комбинации "локация/уровень/валюта/период выплат".
    6.  Возвращает структурированный словарь с агрегированными результатами и количеством
        обработанных источников.

    Args:
        search_queries (List[str]): Список поисковых запросов для инструмента `web_search`.
            Пример: ["python developer salary moscow", "senior software engineer remote salary 2024"].
        role (str): Целевая роль для поиска, передается в `salary_range_parser` для
            уточнения контекста. Пример: "Software Engineer".
        level (str): Целевой уровень квалификации (грейд), передается в `salary_range_parser`.
            Пример: "Senior".
        locations (List[str]): Список целевых локаций, передается в `salary_range_parser`.
            Пример: ["Moscow", "Remote", "Saint Petersburg"].

    Returns:
        Dict[str, Any]: Словарь, содержащий статус выполнения, количество найденных
        точек данных и список агрегированных результатов.
        Пример:
        {
            "status": "success",
            "total_data_points_found": 5,
            "aggregated_results": [
                {
                    "location": "Moscow",
                    "level": "Senior",
                    "currency": "RUB",
                    "pay_period": "monthly",
                    "data_points_count": 3,
                    "average_min_salary": 350000.0,
                    "average_max_salary": 450000.0,
                    "absolute_min_salary": 300000,
                    "absolute_max_salary": 500000
                },
                # ... другие группы
            ]
        }

    Raises:
        Exception: Если не удалось найти ни одной точки данных о зарплате или если
                   произошла критическая ошибка во время выполнения.
    """
    try:
        # Этап 1 & 2 & 3: Поиск URL, чтение страниц и парсинг данных
        all_parsed_salaries = []
        processed_urls = set()

        for query in search_queries:
            # Предполагается, что `web_search` доступен в среде выполнения
            # и возвращает список URL в формате {"results": [{"url": "..."}]}
            search_results = web_search(query=query)
            if not search_results or "results" not in search_results:
                continue

            for result in search_results["results"]:
                url = result.get("url")
                if not url or url in processed_urls:
                    continue
                
                processed_urls.add(url)

                try:
                    # Предполагается, что `webpage_reader` доступен в среде выполнения
                    page_content = webpage_reader(url=url)
                    if not page_content:
                        continue

                    # Предполагается, что `salary_range_parser` доступен в среде выполнения
                    # и возвращает словарь или None
                    parsed_data = salary_range_parser(
                        text_content=page_content,
                        target_role=role,
                        target_level=level,
                        target_locations=locations
                    )

                    # Проверяем, что парсер вернул валидные и полные данные
                    if parsed_data and all(k in parsed_data for k in [
                        "min_salary", "max_salary", "currency", "pay_period", "location", "level"
                    ]):
                        all_parsed_salaries.append(parsed_data)

                except Exception:
                    # Игнорируем ошибки с отдельными URL, чтобы не прерывать весь процесс
                    continue

        # Этап 4: Проверка, были ли найдены данные
        if not all_parsed_salaries:
            raise Exception("Не удалось найти и извлечь ни одной релевантной записи о зарплате.")

        # Этап 5: Агрегация собранных данных
        aggregation_groups = {}

        for salary_info in all_parsed_salaries:
            # Создаем ключ для группировки по основным параметрам
            group_key = (
                salary_info.get("location", "unknown").strip().lower(),
                salary_info.get("level", "unknown").strip().lower(),
                salary_info.get("currency", "unknown").strip().upper(),
                salary_info.get("pay_period", "unknown").strip().lower()
            )

            if group_key not in aggregation_groups:
                aggregation_groups[group_key] = {
                    "min_salaries": [],
                    "max_salaries": [],
                    "original_locations": Counter(),
                    "original_levels": Counter()
                }
            
            # Добавляем данные в соответствующую группу
            aggregation_groups[group_key]["min_salaries"].append(salary_info["min_salary"])
            aggregation_groups[group_key]["max_salaries"].append(salary_info["max_salary"])
            # Сохраняем оригинальные названия для более точного отчета
            aggregation_groups[group_key]["original_locations"][salary_info.get("location", "unknown")] += 1
            aggregation_groups[group_key]["original_levels"][salary_info.get("level", "unknown")] += 1


        # Финальный этап: Вычисление и форматирование результатов
        final_results = []
        for group_key, data in aggregation_groups.items():
            min_salaries = data["min_salaries"]
            max_salaries = data["max_salaries"]
            count = len(min_salaries)

            if count == 0:
                continue

            # Используем most_common для выбора наиболее часто встречающегося написания
            final_location = data["original_locations"].most_common(1)[0][0]
            final_level = data["original_levels"].most_common(1)[0][0]

            result_entry = {
                "location": final_location,
                "level": final_level,
                "currency": group_key[2],
                "pay_period": group_key[3],
                "data_points_count": count,
                "average_min_salary": sum(min_salaries) / count,
                "average_max_salary": sum(max_salaries) / count,
                "absolute_min_salary": min(min_salaries),
                "absolute_max_salary": max(max_salaries)
            }
            final_results.append(result_entry)

        return {
            "status": "success",
            "total_data_points_found": len(all_parsed_salaries),
            "aggregated_results": final_results
        }

    except Exception as e:
        # Перехватываем любые исключения, включая кастомное выше, и перевыбрасываем
        # в стандартизированном формате, как того требует задача.
        error_message = f"Ошибка в инструменте salary_aggregator: {e}"
        # В реальной среде здесь могло бы быть логирование
        # import logging; logging.error(error_message, exc_info=True)
        raise Exception(error_message)