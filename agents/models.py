# agents/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

# --- Модель для Базы Знаний ---
class KnowledgeUnit(BaseModel):
    """
    Структура для одной единицы знания (факта) с версионированием.
    Это "атом" нашей Базы Знаний.
    """
    claim_id: str = Field(description="Уникальный, читаемый ID факта (например, 'moodle_performance_issues_2024').")
    statement: str = Field(description="Четкое, атомарное утверждение.")
    version: int = Field(default=1, description="Версия факта, увеличивается при обновлении.")
    created_at: str = Field(description="Дата и время создания в формате ISO 8601.")
    status: Literal['ACTIVE', 'ARCHIVED'] = Field(default='ACTIVE', description="Статус факта.")
    source_link: str = Field(description="Прямая ссылка на источник.")
    source_quote: str = Field(description="Прямая цитата из источника, подтверждающая утверждение.")

# --- Модели для Рабочих Агентов (Worker Agents) ---
class FactExtractionReport(BaseModel):
    """Отчет от Researcher/Contrarian агентов."""
    extracted_facts: List[KnowledgeUnit] = Field(description="Список извлеченных фактов из проанализированных источников.")

class QualityAssessment(BaseModel):
    """Вердикт контролера качества по одному факту."""
    claim_id: str = Field(description="ID проверяемого факта.")
    is_ok: bool = Field(description="True, если факт качественный и не требует доработки.")
    is_fixable: bool = Field(description="True, если факт имеет недостатки, но их можно исправить (серая зона).")
    reason: str = Field(description="Краткое объяснение, почему факт требует исправления или является браком.")

class BatchQualityAssessmentReport(BaseModel):
    """Пакетный отчет контролера качества."""
    assessments: List[QualityAssessment] = Field(description="Список оценок для каждого факта в пакете.")

# --- Модели для Аналитических и Мета-Агентов ---
class AnalystReport(BaseModel):
    """Структурированный отчет от AnalystAgent для рефлексии."""
    key_insights: List[str] = Field(description="Список из 3-5 ключевых выводов.")
    data_gaps: List[str] = Field(description="Список из 2-3 обнаруженных пробелов в данных.")

class JanitorReport(BaseModel):
    """Отчет от KnowledgeJanitorAgent."""
    conflicts_found: List[List[str]] = Field(description="Список групп ID конфликтующих фактов.")
    archived_ids: List[str] = Field(description="Список ID устаревших фактов, которые следует заархивировать.")

class FinalReport(BaseModel):
    """Финальный отчет от ReportWriterAgent."""
    markdown_content: str = Field(description="Полностью готовый отчет в формате Markdown.")

# --- Модель для SupervisorAgent ---
class GraphPlan(BaseModel):
    """Pydantic-модель для описания плана графа."""
    tasks: List[Dict] = Field(description="Список всех задач, которые нужно выполнить (например, {'task_id': 'res_01', 'agent_name': 'Researcher', 'description': '...'})")
    initial_model_assignments: Dict[str, str] = Field(description="Словарь {task_id: model_name} с начальным распределением моделей.")