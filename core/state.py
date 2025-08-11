# core/state.py
from typing import TypedDict, List, Dict, Any, Optional

# Определяем структуру состояния графа с помощью TypedDict для простоты и производительности.
# Это замена бинарному файлу, обеспечивающая читаемость.
class GraphState(TypedDict):
    """
    Центральное состояние, управляющее всем потоком вычислений.

    Attributes:
        user_config: Конфигурация, загруженная из config.yaml.
        task_queue: Список задач, ожидающих выполнения.
        completed_tasks: Список выполненных задач.
        knowledge_base: База Знаний, словарь с фактами.
        artifacts: Словарь для хранения финальных бизнес-артефактов.
        data_bus: Шина данных для передачи сырых результатов между задачами.
        model_assignments: Распределение моделей по задачам.
        visited_urls: Список URL, которые уже были посещены.
        escalation_count: Счетчик эскалаций для одной задачи.
        current_task: Текущая выполняемая задача.
        error_message: Сообщение об ошибке для последней неудачи.
        node_outputs: Внутреннее хранилище для результатов узлов.
        report_outline: План (оглавление) для финального отчета.
        drafted_sections: Список написанных секций отчета.
        current_section_to_draft: Текущая секция, над которой идет работа.
    """
    user_config: Dict
    task_queue: List[Dict]
    completed_tasks: List[Dict]
    knowledge_base: Dict[str, Any]
    artifacts: Dict[str, Any]
    # --- ИЗМЕНЕНИЕ НАЧАТО: Добавлена шина данных ---
    data_bus: Dict[str, Any]
    # --- ИЗМЕНЕНИЕ ОКОНЧЕНО ---
    model_assignments: Dict[str, str]
    visited_urls: List[str]
    escalation_count: int
    current_task: Optional[Dict]
    error_message: Optional[str]
    node_outputs: Dict[str, Any]
    report_outline: Dict
    drafted_sections: List[Dict]
    current_section_to_draft: Optional[Dict]