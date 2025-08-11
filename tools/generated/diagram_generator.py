import json
from typing import Dict, Any, List

# Разрешенные импорты, которые не используются в данной реализации,
# но могут быть использованы для будущих расширений (например, для получения данных по URL).
import requests
from bs4 import BeautifulSoup


def diagram_generator(data_json: str) -> Dict[str, Any]:
    """
    Генерирует код диаграммы Ганта в формате Mermaid.js на основе структурированных данных.

    Эта функция спроектирована как надежный и изолированный компонент. Она не пытается
    интерпретировать естественный язык или извлекать данные из внешних источников. Вместо этого
    она принимает на вход строго структурированные данные в формате JSON и преобразует их
    в текстовое представление диаграммы. Такой подход обеспечивает предсказуемость,
    тестируемость и независимость от внешних сервисов.

    Ожидаемая структура JSON-строки в аргументе `data_json`:
    {
      "title": "Название Проекта",
      "dateFormat": "YYYY-MM-DD", // Формат даты для Mermaid.js
      "tasks": [
        {
          "id": "task1", // Уникальный идентификатор
          "name": "Проектирование",
          "start_date": "2023-01-01",
          "end_date": "2023-01-15",
          "status": "done" // Опционально: "done", "active", "crit"
        },
        {
          "id": "task2",
          "name": "Разработка",
          "start_date": "2023-01-16",
          "end_date": "2023-02-28",
          "depends_on": "task1" // Опционально: зависимость от другой задачи
        },
        {
          "name": "Тестирование", // id опционален, будет сгенерирован из имени
          "start_date": "2023-03-01",
          "duration": "10d" // Можно указать длительность вместо end_date
        }
      ],
      "sections": [ // Опционально: для группировки задач
        {
          "name": "Этап 1: Планирование",
          "task_ids": ["task1"]
        },
        {
          "name": "Этап 2: Реализация",
          "task_ids": ["task2"]
        }
      ]
    }

    Args:
        data_json: Строка в формате JSON, содержащая данные для построения диаграммы.

    Returns:
        Словарь, содержащий статус операции и сгенерированный код диаграммы.
        Пример: {"status": "success", "diagram_type": "mermaid", "code": "gantt\n..."}

    Raises:
        Exception: Если входная строка не является валидным JSON, если структура
                   JSON не соответствует ожиданиям, или если возникают другие ошибки
                   при генерации кода диаграммы.
    """
    try:
        # Шаг 1: Парсинг и валидация входных данных
        try:
            data: Dict[str, Any] = json.loads(data_json)
        except json.JSONDecodeError as e:
            raise Exception(f"Ошибка декодирования JSON: {e}. Убедитесь, что передана валидная JSON-строка.")

        # Проверка наличия обязательных полей
        if "tasks" not in data or not isinstance(data["tasks"], list):
            raise ValueError("В JSON отсутствует обязательный ключ 'tasks' или его значение не является списком.")
        if not data["tasks"]:
            raise ValueError("Список 'tasks' не может быть пустым.")

        # Шаг 2: Генерация кода диаграммы Mermaid.js
        mermaid_code: List[str] = ["gantt"]

        # Добавление заголовка и формата даты
        title: str = data.get("title", "Диаграмма Ганта")
        date_format: str = data.get("dateFormat", "YYYY-MM-DD")
        mermaid_code.append(f"    title {title}")
        mermaid_code.append(f"    dateFormat {date_format}")

        # Обработка задач и секций
        tasks: List[Dict[str, Any]] = data["tasks"]
        sections: List[Dict[str, Any]] = data.get("sections", [])
        
        processed_task_ids = set()

        if sections:
            for section in sections:
                section_name = section.get("name")
                section_task_ids = section.get("task_ids")
                if not section_name or not section_task_ids:
                    continue # Пропускаем некорректно определенные секции
                
                mermaid_code.append(f"    section {section_name}")
                
                for task in tasks:
                    task_id = task.get("id", task.get("name", "").replace(" ", "_"))
                    if task_id in section_task_ids:
                        task_line = _format_task_line(task)
                        mermaid_code.append(f"    {task_line}")
                        processed_task_ids.add(task_id)
        
        # Добавление задач, не вошедших ни в одну секцию
        has_unsectioned_tasks = any(task.get("id", task.get("name", "").replace(" ", "_")) not in processed_task_ids for task in tasks)
        if has_unsectioned_tasks:
            if sections: # Добавляем заголовок для "прочих" задач, если уже были секции
                 mermaid_code.append("    section Прочие задачи")
            for task in tasks:
                task_id = task.get("id", task.get("name", "").replace(" ", "_"))
                if task_id not in processed_task_ids:
                    task_line = _format_task_line(task)
                    mermaid_code.append(f"    {task_line}")

        # Шаг 3: Формирование и возврат результата
        final_code = "\n".join(mermaid_code)
        return {
            "status": "success",
            "diagram_type": "mermaid",
            "code": final_code
        }

    except (ValueError, KeyError, TypeError) as e:
        # Обработка ошибок, связанных с неверной структурой или содержимым JSON
        raise Exception(f"Ошибка в структуре данных: {e}")
    except Exception as e:
        # Перехват всех остальных непредвиденных ошибок
        raise Exception(f"Неизвестная ошибка при генерации диаграммы: {e}")


def _format_task_line(task: Dict[str, Any]) -> str:
    """Вспомогательная функция для форматирования одной строки задачи для Mermaid.js."""
    name = task.get("name")
    if not name:
        raise ValueError("У каждой задачи должно быть поле 'name'.")

    # Обязательные поля для определения временного интервала
    start_date = task.get("start_date")
    end_date = task.get("end_date")
    duration = task.get("duration")

    if not start_date:
        raise ValueError(f"Задача '{name}' не имеет обязательного поля 'start_date'.")
    if not end_date and not duration:
        raise ValueError(f"Задача '{name}' должна иметь либо 'end_date', либо 'duration'.")

    # Формирование строки
    line_parts = [f"{name} "]
    
    # Статус задачи
    status = task.get("status")
    if status in ["done", "active", "crit"]:
        line_parts.append(f":{status}, ")

    # ID задачи и зависимости
    task_id = task.get("id")
    depends_on = task.get("depends_on")
    if task_id or depends_on:
        # Если есть зависимость, ID обязателен
        if depends_on and not task_id:
            raise ValueError(f"Задача '{name}' имеет зависимость, но не имеет 'id'.")
        
        # Используем id или генерируем его из имени для ссылок
        effective_id = task_id if task_id else name.replace(" ", "_")
        line_parts.append(f"{effective_id}, ")

    # Временной интервал
    line_parts.append(f"{start_date}, ")
    if end_date:
        line_parts.append(end_date)
    else: # duration
        line_parts.append(duration)

    return "".join(line_parts)