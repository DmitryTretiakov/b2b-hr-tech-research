# core/state.py
from typing import List, Dict, TypedDict, Optional

# --- Модели данных ---
class KnowledgeUnit(TypedDict):
    """Структура для одной единицы знания (факта) с версионированием."""
    claim_id: str
    statement: str
    version: int
    created_at: str
    status: str  # 'ACTIVE', 'ARCHIVED'
    source_link: str
    # ... другие поля факта

class Task(TypedDict):
    """Структура для одной задачи."""
    task_id: str
    description: str
    agent_name: str # e.g., 'Researcher', 'Janitor'
    status: str # 'PENDING', 'COMPLETED', 'FAILED'
    
# --- Главное Состояние Графа ---
class GraphState(TypedDict):
    """
    Представляет полное состояние нашего графа.
    Передается между всеми узлами.
    """
    # Контекст, загружаемый при старте
    user_config: Dict

    # Динамически изменяемые поля
    task_queue: List[Task]
    completed_tasks: List[Task]
    knowledge_base: Dict[str, KnowledgeUnit]
    artifacts: Dict[str, Dict]  # <-- НОВОЕ ПОЛЕ ДЛЯ ХРАНЕНИЯ АРТЕФАКТОВ
    visited_urls: List[str]
    
    # Поля для управления эскалацией
    current_task: Optional[Task]
    model_assignments: Dict[str, str]
    escalation_count: int
    
    # Поля для передачи результатов между узлами
    error_message: Optional[str]
    node_outputs: Dict[str, List]

    # Поля для многоэтапного написания отчета
    report_outline: Optional[Dict]
    drafted_sections: List[Dict]
    current_section_to_draft: Optional[Dict]