# agents/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional


class ValidationReport(BaseModel):
    """
    Структурированный отчет от ValidatorAgent.
    Определяет, можно ли выполнить задачу и как именно.
    """
    is_executable: bool = Field(description="True, если задача полностью выполнима с помощью предоставленных инструментов.")
    reasoning: str = Field(description="Краткое объяснение, почему задача выполнима или невыполнима.")
    missing_tool_description: Optional[str] = Field(default=None, description="Если is_executable=false, здесь должно быть четкое описание того, какой инструмент необходимо создать.")
    # === ИЗМЕНЕНИЕ НАЧАТО: Добавлено поле для плана ===
    suggested_plan: Optional[List[str]] = Field(default=None, description="Если задача сложная, но выполнимая, здесь должен быть пошаговый план для исполнителя.")

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

class SanityCheckReport(BaseModel):
    """Отчет от SanityCheckCritic, содержащий ID проверенных фактов."""
    verified_claim_ids: List[str] = Field(description="Список ID фактов, которые успешно прошли финальную проверку на здравый смысл и релевантность.")

# --- Модели для Аналитических и Мета-Агентов ---
class AnalystReport(BaseModel):
    """Структурированный отчет от AnalystAgent для рефлексии."""
    key_insights: List[str] = Field(description="Список из 3-5 ключевых выводов.")
    data_gaps: List[str] = Field(description="Список из 2-3 обнаруженных пробелов в данных.")

class FinalAnalysisReport(BaseModel):
    """Структурированные данные для финального отчета, сгенерированные AnalystAgent."""
    title: str = Field(description="Главный заголовок всего отчета.")
    executive_summary: str = Field(description="Краткая выжимка (Executive Summary) на 2-3 абзаца.")
    key_findings: List[Dict] = Field(description="Список ключевых находок, где каждый элемент - это словарь с ключами 'finding' (формулировка находки) и 'supporting_claim_ids' (список ID подтверждающих фактов).")
    conclusion: str = Field(description="Финальное заключение, обобщающее результаты.")
    recommendations: List[str] = Field(description="Список конкретных рекомендаций на основе анализа.")

class JanitorReport(BaseModel):
    """Отчет от KnowledgeJanitorAgent."""
    conflicts_found: List[List[str]] = Field(description="Список групп ID конфликтующих фактов.")
    archived_ids: List[str] = Field(description="Список ID устаревших фактов, которые следует заархивировать.")

class FinalReport(BaseModel):
    """Финальный отчет от ReportWriterAgent."""
    markdown_content: str = Field(description="Полностью готовый отчет в формате Markdown.")

class ArchitectDecision(BaseModel):
    """Модель для валидации решения ArchitectAgent."""
    action: Literal['FIX_DESCRIPTION', 'CREATE_TOOL'] = Field(description="Выбранное действие: исправить описание или создать инструмент.")
    data: Dict[str, str] = Field(description="Данные для выполнения действия. Для 'FIX_DESCRIPTION' содержит {'new_description': '...'}. Для 'CREATE_TOOL' содержит {'tool_name': '...', 'tool_description': '...'}.")

# --- Модели для Генерации Артефактов ---
class FinancialModelArtifact(BaseModel):
    """Структурированный артефакт для базовой финансовой модели."""
    title: str = Field(description="Название финансовой модели, например, 'Прогноз юнит-экономики для MVP'.")
    key_assumptions: List[str] = Field(description="Список ключевых допущений, на которых построена модель (например, 'Средний чек (ACV) = 300,000 руб/год').")
    calculations_table_markdown: str = Field(description="Таблица с расчетами, отформатированная как Markdown. Должна включать основные метрики (LTV, CAC, OPEX, CAPEX).")
    summary_conclusion: str = Field(description="Краткий вывод по результатам моделирования (например, 'Модель показывает положительную юнит-экономику на второй год при удержании клиента > 18 месяцев.').")

class UserStory(BaseModel):
    """Структура для одной User Story."""
    role: str = Field(description="Роль пользователя, например, 'HR-директор'.")
    action: str = Field(description="Действие, которое пользователь хочет совершить, например, 'видеть прогресс обучения сотрудников по ключевым компетенциям'.")
    value: str = Field(description="Ценность, которую пользователь получает, например, 'чтобы принимать решения о кадровых перестановках'.")

class UserStoryArtifact(BaseModel):
    """Структурированный артефакт для набора User Stories, формирующих дорожную карту."""
    epic_title: str = Field(description="Название верхнеуровневой задачи (эпика), например, 'Реализация MVP Карьерного Навигатора'.")
    user_stories: List[UserStory] = Field(description="Список User Stories, детализирующих эпик.")

# --- Модель для SupervisorAgent ---
class GraphPlan(BaseModel):
    """Pydantic-модель для описания плана графа."""
    tasks: List[Dict] = Field(description="Список всех задач, которые нужно выполнить (например, {'task_id': 'res_01', 'agent_name': 'Researcher', 'description': '...'})")
    initial_model_assignments: Dict[str, str] = Field(description="Словарь {task_id: model_name} с начальным распределением моделей.")

class RevisionReport(BaseModel):
    """Модель для вердикта ReviserAgent."""
    is_sufficient: bool = Field(description="True, если собранной информации достаточно и можно переходить к следующему этапу (QA).")
    feedback: str = Field(description="Конструктивная критика и конкретные предложения по следующим шагам, если информация неполна. Например, 'Собраны данные только по рынку РФ, необходимо исследовать рынок СНГ.'")
    new_task_suggestions: List[str] = Field(description="Список формулировок для новых исследовательских задач, если они необходимы.")

class ReportOutline(BaseModel):
    """Структура для плана (оглавления) финального отчета."""
    title: str = Field(description="Главный заголовок всего отчета.")
    sections: List[Dict[str, str]] = Field(description="Список секций отчета, где каждый элемент - это словарь с ключами 'section_title' и 'section_description' (о чем писать в этой секции).")

class ReportSection(BaseModel):
    """Структура для текста одной секции отчета."""
    markdown_content: str = Field(description="Полностью написанный текст для одной секции в формате Markdown.")


# 1. Модель для ProductOwnerMemoAgent и InvestmentMemoAgent
class MemoArtifact(BaseModel):
    """Структурированный артефакт для аналитической или инвестиционной записки."""
    title: str = Field(description="Главный заголовок документа.")
    executive_summary: str = Field(description="Краткая выжимка (Executive Summary) на 2-3 абзаца, излагающая суть и ключевые выводы.")
    markdown_content: str = Field(description="Основное тело записки в формате Markdown. Должно быть структурировано с подзаголовками.")
    recommendation: str = Field(description="Четкая и однозначная рекомендация (например, 'Рекомендуется приступить к разработке MVP' или 'Рекомендуется провести дополнительное исследование рынка').")

# 2. Модель для CompetitorAnalysisAgent
class CompetitorProfile(BaseModel):
    """Профиль одного конкурента."""
    name: str = Field(description="Название компании-конкурента.")
    strengths: List[str] = Field(description="Список из 2-3 ключевых сильных сторон.")
    weaknesses: List[str] = Field(description="Список из 2-3 ключевых слабых сторон.")
    business_model: str = Field(description="Краткое описание бизнес-модели (например, 'SaaS-подписка по tiered-модели').")

class CompetitorAnalysisArtifact(BaseModel):
    """Структурированный артефакт для анализа конкурентов."""
    market_overview: str = Field(description="Краткий обзор рынка (1-2 абзаца).")
    competitors: List[CompetitorProfile] = Field(description="Список профилей ключевых конкурентов.")
    strategic_conclusion: str = Field(description="Стратегический вывод о положении нашей компании на фоне конкурентов.")

# 3. Модель для TechnologyDeepDiveAgent
class TechnologyAssessment(BaseModel):
    """Оценка одного технологического аспекта."""
    aspect: str = Field(description="Анализируемый аспект (например, 'Риски производительности Moodle').")
    assessment: str = Field(description="Оценка аспекта, включая потенциальные риски и возможности (2-3 предложения).")
    recommendation: str = Field(description="Конкретная рекомендация (например, 'Провести нагрузочное тестирование перед запуском' или 'Рассмотреть альтернативные LMS для B2B-сегмента').")

class TechnologyDeepDiveArtifact(BaseModel):
    """Структурированный артефакт для глубокого технического анализа."""
    overall_summary: str = Field(description="Общий вывод о технологической готовности и ключевых рисках.")
    assessments: List[TechnologyAssessment] = Field(description="Список оценок по конкретным техническим аспектам.")

# 4. Модель для RoadmapVisualizationAgent
class RoadmapVisualizationArtifact(BaseModel):
    """Артефакт для визуализации дорожной карты."""
    title: str = Field(description="Заголовок дорожной карты.")
    mermaid_diagram: str = Field(description="Полностью готовая диаграмма в синтаксисе Mermaid.js, представляющая дорожную карту (например, диаграмма Ганта).")
    commentary: str = Field(description="Краткий комментарий, объясняющий ключевые этапы на диаграмме.")