# agents/models.py
from pydantic import BaseModel, Field
from typing import List, Literal, Dict

# Модели для планирования
class Task(BaseModel):
    """Описывает одну конкретную задачу в рамках фазы проекта."""
    task_id: str = Field(description="Уникальный идентификатор задачи, например 'task_001'. Должен быть уникальным во всем плане.")
    # ИЗМЕНЕНО: Добавлены новые типы агентов
    assignee: Literal[
        'ResearcherAgent', 'ContrarianAgent',
        'ProductOwnerAgent', 'FinancialModelAgent', 'ProductManagerAgent',
        'System_Diagnostician'
    ] = Field(description="Эксперт, которому поручена задача.")
    description: str = Field(description="Четкое и краткое описание задачи для эксперта.")
    goal: str = Field(description="Бизнес-цель, на которую направлена эта задача. Что мы хотим узнать?")
    status: Literal['PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED'] = Field(description="Текущий статус задачи. Новые задачи всегда PENDING.")
    retry_count: int = Field(default=0, description="Счетчик повторных попыток выполнения задачи в случае сбоя API.")
    # НОВОЕ ПОЛЕ: Для связи состязательных задач
    pair_id: str = Field(default="", description="Общий ID для состязательной пары задач (Researcher/Contrarian).")


class Phase(BaseModel):
    """Описывает одну фазу проекта, состоящую из нескольких задач."""
    phase_name: str = Field(description="Название фазы, например 'Phase 1: Глубокая Разведка Активов ТГУ'.")
    status: Literal['PENDING', 'IN_PROGRESS', 'COMPLETED'] = Field(description="Текущий статус фазы.")
    tasks: List[Task] = Field(description="Список задач для этой фазы.")

class StrategicPlan(BaseModel):
    """Описывает полный стратегический план проекта."""
    main_goal_status: Literal['IN_PROGRESS', 'READY_FOR_FINAL_BRIEF', 'FAILED'] = Field(description="Общий статус всего проекта. IN_PROGRESS, пока идет работа.")
    phases: List[Phase] = Field(description="Список всех фаз проекта.")

# Модели для генерации данных
class Claim(BaseModel):
    """Описывает одно конкретное, верифицируемое 'Утверждение' (Claim)."""
    claim_id: str = Field(description="Короткий, уникальный и информативный ID на английском (например, 'websoft_customization_weakness').")
    statement: str = Field(description="Четкий вывод или факт, сформулированный как утверждение.")
    value: str = Field(description="Конкретное значение (цифра, текст, список), подтверждающее утверждение.")
    source_link: str = Field(description="Прямая ссылка на самый релевантный источник, откуда взята информация.")
    source_quote: str = Field(description="Прямая цитата из источника, которая доказывает утверждение.")
    confidence_score: float = Field(description="Оценка уверенности в достоверности источника от 0.0 до 1.0.")
    status: Literal["UNVERIFIED", "VERIFIED", "CONFLICTED", "DEPRECATED"] = Field(description="Статус утверждения. На этапе создания всегда 'UNVERIFIED'.")
    source_type: str = Field(default="UNKNOWN", description="Тип источника (например, OFFICIAL_DOCS, FORUM_POST).")
    source_trust: float = Field(default=0.2, description="Коэффициент доверия к самому ИСТОЧНИКУ, от 0.0 до 1.0.")

class ClaimList(BaseModel):
    """Описывает список 'Утверждений'."""
    claims: List[Claim]

class SearchQueries(BaseModel):
    """Описывает список поисковых запросов."""
    queries: List[str] = Field(description="Список из 4-6 конкретных и разнообразных поисковых запросов на русском языке.")

# Модели для контроля качества
class QualityAssessment(BaseModel):
    """Описывает вердикт контролера качества по одному утверждению."""
    claim_id: str = Field(description="ID проверяемого утверждения.")
    is_ok: bool = Field(description="True, если утверждение качественное и не требует доработки.")
    is_fixable: bool = Field(description="True, если утверждение имеет недостатки, но их можно исправить (серая зона).")
    reason: str = Field(description="Краткое объяснение, почему утверждение требует исправления или является браком.")

class BatchQualityAssessmentReport(BaseModel):
    """Пакетный отчет контролера качества."""
    assessments: List[QualityAssessment] = Field(description="Список оценок для каждого утверждения в пакете.")

# Модели для синтеза и отчетов
class AnalystReport(BaseModel):
    """Структурированный отчет от Аналитика для Супервайзера."""
    key_insights: List[str] = Field(description="Список из 3-5 ключевых выводов, сделанных на основе текущей Базы Знаний.")
    data_gaps: List[str] = Field(description="Список из 2-3 обнаруженных пробелов в данных, требующих дополнительного исследования.")
    confidence_level: float = Field(description="Общая оценка уверенности в полноте данных для текущей фазы (от 0.0 до 1.0).")

class FinalReport(BaseModel):
    """Модель для финального отчета, который пишет ReportWriter."""
    markdown_content: str = Field(description="Полностью готовый отчет в формате Markdown.")

class ValidationReport(BaseModel):
    """Описывает результат проверки артефакта на качество."""
    is_valid: bool = Field(description="True, если артефакт прошел все проверки, иначе False.")
    reasons: List[str] = Field(description="Список конкретных причин, по которым артефакт был признан невалидным. Пустой, если is_valid=True.")