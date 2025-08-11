# orchestrator.py
from langgraph.graph import StateGraph, END
from core.state import GraphState
from agents.supervisor import SupervisorAgent
from agents.workers import (
    ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, 
    AnalystAgent, ReportWriterAgent, SanityCheckCritic,
    FinancialModelAgent, ProductManagerAgent, ReviserAgent,
    OutlineAgent, SectionWriterAgent
)
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ValidatorAgent, FailureAnalystAgent
from utils.helpers import citation_post_processor
import os
import json
import traceback

# --- Константы для эскалации ---
MAX_ESCALATIONS_PER_TASK = 2
MODEL_ESCALATION_PATH = {
    "gemma-3": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
    "gemini-2.5-flash": None
}

# ====================================================================================
# === 1. ОПРЕДЕЛЕНИЕ УЗЛОВ ГРАФА (NODES) =============================================
# ====================================================================================
def final_audit_node(state: GraphState, architect: ArchitectAgent) -> GraphState:
    """
    Вызывает архитектора для финальной проверки состояния системы.
    """
    print("\n--- Узел: Final Audit ---")
    audit_report = architect.conduct_final_audit(state)
    state.setdefault('node_outputs', {})['final_audit_report'] = audit_report
    return state

def supervisor_node(state: GraphState, supervisor: SupervisorAgent) -> GraphState:
    print("\n--- Узел: Supervisor ---")
    if not state.get('task_queue') and not state.get('completed_tasks'):
        print("   [SupervisorNode] Генерирую первоначальный план на основе user_config...")
        plan = supervisor.create_initial_plan(state['user_config'])
        state['task_queue'].extend(plan.get('tasks', []))
        state['model_assignments'].update(plan.get('initial_model_assignments', {}))
        state['node_outputs'] = {'accumulated_raw_facts': []}
    return state

def task_fetcher_node(state: GraphState) -> GraphState:
    """
    Берет задачу из очереди. НЕ сбрасывает счетчик эскалации.
    """
    print("\n--- Узел: Task Fetcher ---")
    if not state['task_queue']: return state
    task = state['task_queue'].pop(0)
    state['current_task'] = task
    state['error_message'] = None
    input_data = {}
    data_dependencies = task.get('data_dependencies', [])
    if data_dependencies:
        print(f"   [FetcherNode] -> Собираю данные для зависимостей: {data_dependencies}")
        for dep_task_id in data_dependencies:
            if dep_task_id in state['data_bus']:
                input_data[dep_task_id] = state['data_bus'][dep_task_id]
            else:
                # Этого не должно случиться при правильном планировании, но на всякий случай
                print(f"   [FetcherNode] !!! ВНИМАНИЕ: Зависимость по данным '{dep_task_id}' не найдена в data_bus!")
    # Помещаем собранные данные в саму задачу для легкого доступа агентом
    task['input_data'] = input_data
    # Также передаем всю Базу Знаний для интеллектуальных агентов
    task['input_data']['knowledge_base'] = state.get('knowledge_base', {})

    print(f"   [FetcherNode] -> Взял в работу задачу: {task['task_id']} ({task['agent_name']})")
    return state

def tool_validator_node(state: GraphState, validator: ValidatorAgent) -> GraphState:
    """
    Проверяет, можно ли выполнить задачу. Пропускает "интеллектуальные" задачи напрямую.
    """
    print("\n--- Узел: Tool Validator ---")
    task = state.get('current_task')
    if not task: return state

    # Список агентов, которые не используют внешние инструменты, а работают за счет LLM.
    INTELLECTUAL_AGENTS = [
        "CompetitorAnalysisAgent",
        "TechnologyDeepDiveAgent",
        "ProductOwnerMemoAgent",
        "InvestmentMemoAgent",
        "FinancialModelAgent", # Хотя он генерирует артефакт, он делает это на основе KB, а не инструментов
        "ProductManagerAgent",
        "OutlineAgent",
        "SectionWriterAgent",
        "ReportWriterAgent"
    ]

    agent_name = task.get('agent_name')
    if agent_name in INTELLECTUAL_AGENTS:
        print(f"   [ValidatorNode] -> Задача для интеллектуального агента '{agent_name}'. Пропускаю напрямую к исполнителю.")
        # Создаем "пустой" положительный отчет, чтобы маршрутизатор сработал правильно.
        report = {"is_executable": True, "reasoning": "Задача для интеллектуального агента, инструменты не требуются."}
    else:
        print(f"   [ValidatorNode] -> Задача для инструментального агента '{agent_name}'. Запускаю полную проверку.")
        report = validator.execute(task)
    
    state.setdefault('node_outputs', {})['validation_report'] = report
    return state


def task_executor_node(state: GraphState, agents: dict) -> GraphState:
    """
    Выполняет одну задачу с детальным логированием ошибок.
    """
    print("\n--- Узел: Task Executor ---")
    task = state.get('current_task')
    if not task: return state
    agent_name = task['agent_name']
    agent = agents.get(agent_name)
    model = state['model_assignments'].get(task['task_id'])

    if not agent or not model:
        task['status'] = 'FAILURE'
        state['error_message'] = f"Агент {agent_name} или модель не найдены."
    else:
        try:
            # Агенты теперь получают все состояние, чтобы иметь доступ к user_config и другим мета-данным
            result = agent.execute(task, model, state)

            if result is None:
                task['status'] = 'FAILURE'
                state['error_message'] = "Агент не вернул результат (None). Вероятная причина - внутренняя ошибка агента."
            else:
                task['status'] = 'SUCCESS'
                # --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: Результат кладется в data_bus ---
                state.setdefault('data_bus', {})[task['task_id']] = result
                print(f"   [ExecutorNode] -> Результат задачи '{task['task_id']}' записан в data_bus.")

        # === ИЗМЕНЕНИЕ НАЧАТО: Заменено на BaseException для перехвата абсолютно всех ошибок ===
        except BaseException as e:
            print("\n" + "="*80)
            print(f"!!! [Task Executor] ПЕРЕХВАЧЕНА КРИТИЧЕСКАЯ ОШИБКА (BaseException) при выполнении задачи '{task.get('task_id')}' агентом '{agent_name}'.")
            print(f"    Тип ошибки: {type(e).__name__}")
            print(f"    Сообщение об ошибке: {e}")
            traceback.print_exc()
            print("="*80 + "\n")
            task['status'] = 'FAILURE'
            state['error_message'] = f"Критический сбой: {type(e).__name__}: {e}"
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===
    
    state['completed_tasks'].append(task)
    state['current_task'] = None
    return state

def qa_node(state: GraphState, assessor: QualityAssessorAgent) -> GraphState:
    print("\n--- Узел: Quality Assessment ---")
    facts_to_assess = state['node_outputs'].get('facts_for_assessment', [])
    if not facts_to_assess:
        state['node_outputs']['good'], state['node_outputs']['fixable'] = [], []
        return state
    model = "gemma-3"
    report = assessor.execute(facts_to_assess, model, state.get('user_config', {}))
    good, fixable = [], []
    assessments = {item['claim_id']: item for item in report.get('assessments', [])}
    for fact in facts_to_assess:
        assessment = assessments.get(fact['claim_id'])
        if assessment and assessment.get('is_ok'): good.append(fact)
        elif assessment and assessment.get('is_fixable'):
            fact['feedback'] = assessment.get('reason')
            fixable.append(fact)
    state['node_outputs']['good'], state['node_outputs']['fixable'] = good, fixable
    return state

def fixer_node(state: GraphState, fixer: FixerAgent) -> GraphState:
    print("\n--- Узел: Fixer ---")
    facts_to_fix = state['node_outputs'].get('fixable', [])
    if not facts_to_fix:
        state['node_outputs']['fixed'] = []
        return state
    model = "gemma-3"
    fixed_facts = fixer.execute(facts_to_fix, model, state.get('user_config', {}))
    state['node_outputs']['fixed'] = fixed_facts
    return state

def sanity_check_node(state: GraphState, critic: SanityCheckCritic) -> GraphState:
    print("\n--- Узел: Sanity Check ---")
    candidates = state['node_outputs'].get('candidates_for_sanity_check', [])
    if not candidates:
        state['node_outputs']['final_facts'] = []
        return state
    model = "gemini-2.5-pro"
    final_facts = critic.execute(candidates, model, state.get('user_config', {}))
    state['node_outputs']['final_facts'] = final_facts
    return state

def commit_node(state: GraphState) -> GraphState:
    print("\n--- Узел: Commit to KB ---")
    final_facts = state['node_outputs'].get('final_facts', [])
    current_kb = state.get('knowledge_base', {})
    for fact in final_facts: current_kb[fact['claim_id']] = fact
    state['knowledge_base'] = current_kb
    state['node_outputs'] = {'accumulated_raw_facts': []}
    return state

def janitor_node(state: GraphState, janitor: KnowledgeJanitorAgent) -> GraphState:
    print("\n--- Узел: Janitor ---")
    state['knowledge_base'] = janitor.cleanup_knowledge_base(state['knowledge_base'])
    return state

def reflection_node(state: GraphState, supervisor: SupervisorAgent) -> GraphState:
    print("\n--- Узел: Reflection ---")
    # AnalystAgent больше не нужен на этом этапе, его роль выполняет Supervisor
    
    # Вызываем Supervisor для создания плана артефактов и отчета
    next_phase_plan = supervisor.create_artifact_and_report_plan(state)
    
    new_tasks = next_phase_plan.get('tasks', [])
    if new_tasks:
        state['task_queue'].extend(new_tasks)
        state['model_assignments'].update(next_phase_plan.get('initial_model_assignments', {}))
        
    return state

def kb_ingestion_node(state: GraphState) -> GraphState:
    """
    НОВЫЙ УЗЕЛ. Преобразует сырой результат из data_bus в кандидатов для БЗ.
    """
    print("\n--- Узел: KB Ingestion ---")
    last_task = state['completed_tasks'][-1]
    task_id = last_task['task_id']
    raw_result = state.get('data_bus', {}).get(task_id)

    if not raw_result or not isinstance(raw_result, list):
        print(f"   [IngestionNode] -> Нет данных для обработки от задачи {task_id}. Пропускаю.")
        state.setdefault('node_outputs', {})['facts_for_assessment'] = []
        return state

    # Здесь можно добавить сложную логику валидации и преобразования.
    # Пока что мы просто принимаем, что результат - это уже готовый список фактов.
    # Это компромисс, чтобы не усложнять, но в будущем здесь может быть вызов LLM.
    facts_to_assess = raw_result
    print(f"   [IngestionNode] -> Подготовлено {len(facts_to_assess)} фактов-кандидатов для QA.")
    state.setdefault('node_outputs', {})['facts_for_assessment'] = facts_to_assess
    return state

def artifact_commit_node(state: GraphState) -> GraphState:
    """
    НОВЫЙ УЗЕЛ. Перемещает готовый артефакт из data_bus в финальное хранилище.
    """
    print("\n--- Узел: Artifact Commit ---")
    last_task = state['completed_tasks'][-1]
    task_id = last_task['task_id']
    artifact_data = state.get('data_bus', {}).get(task_id)

    if artifact_data:
        state.setdefault('artifacts', {})[task_id] = artifact_data
        print(f"   [ArtifactCommit] -> Артефакт '{task_id}' сохранен.")
    return state
def commit_to_kb_node(state: GraphState) -> GraphState:
    """
    Фиксирует проверенные факты в Базе Знаний.
    """
    print("\n--- Узел: Commit to KB ---")
    final_facts = state['node_outputs'].get('final_facts', [])
    current_kb = state.get('knowledge_base', {})
    for fact in final_facts:
        current_kb[fact['claim_id']] = fact
    state['knowledge_base'] = current_kb
    # Очищаем промежуточные данные
    state['node_outputs'] = {}
    print(f"   [CommitToKB] -> {len(final_facts)} фактов добавлено в Базу Знаний.")
    return state

def failure_analyst_node(state: GraphState, failure_analyst: FailureAnalystAgent) -> GraphState:
    """
    Вызывает агента-диагноста для анализа последней проваленной задачи.
    """
    print("\n--- Узел: Failure Analyst ---")
    # Увеличиваем счетчик эскалаций ПЕРЕД анализом
    state['escalation_count'] = state.get('escalation_count', 0) + 1
    
    last_failed_task = state['completed_tasks'][-1]
    error_message = state['error_message']

    analysis_report = failure_analyst.execute(last_failed_task, error_message, state)
    state.setdefault('node_outputs', {})['failure_analysis_report'] = analysis_report
    return state

def prepare_for_architect_node(state: GraphState) -> GraphState:
    """
    Этот узел-помощник выполняет атомарную операцию: готовит состояние для Архитектора.
    Он берет последнюю проваленную задачу и помещает ее в current_task.
    """
    print("\n--- Узел: Prepare for Architect ---")
    if state['completed_tasks']:
        last_failed_task = state['completed_tasks'][-1]
        state['current_task'] = last_failed_task
        print(f"   [PrepareArchitect] -> Передаю задачу '{last_failed_task['task_id']}' на анализ Архитектору.")
    else:
        print("   [PrepareArchitect] !!! Нет выполненных задач для анализа. Этого не должно было случиться.")
        # Устанавливаем current_task в None, чтобы архитектор корректно обработал ошибку
        state['current_task'] = None
    return state

def architect_node(state: GraphState, architect: ArchitectAgent, validator: ValidatorAgent) -> GraphState:
    """
    Узел для самокоррекции. Теперь работает в цикле до тех пор,
    пока задача не станет полностью выполнимой.
    """
    print("\n--- Узел: Architect ---")
    
    max_iterations = 3
    for i in range(max_iterations):
        print(f"   [ArchitectNode] Итерация самокоррекции {i + 1}/{max_iterations}...")
        
        # На каждой итерации нам нужна "свежая" копия задачи из current_task
        task_to_fix = state.get('current_task')
        if not task_to_fix:
            print("!!! [ArchitectNode] КРИТИЧЕСКАЯ ОШИБКА: Нет текущей задачи для работы.")
            state.setdefault('node_outputs', {})['architect_status'] = 'FATAL_ERROR'
            return state

        # Шаг 1: Проверяем, выполнима ли задача СЕЙЧАС
        validation_report = validator.execute(task_to_fix)
        if validation_report.get('is_executable'):
            print("   [ArchitectNode] <- Задача стала выполнимой. Возвращаю в очередь.")
            state.setdefault('node_outputs', {})['architect_status'] = 'REMEDIATED'
            # Важно: current_task должен быть перемещен обратно в очередь, а не оставаться в current_task
            state['task_queue'].insert(0, state.pop('current_task'))
            state['escalation_count'] = 0
            return state

        # Шаг 2: Если невыполнима, создаем новый инструмент
        failure_feedback = state.get('node_outputs', {}).get('architect_feedback')
        # Мы передаем задачу, но не ожидаем ее изменения, так как статус меняется только при ошибке
        remediated_task_attempt = architect.fix_or_enhance(task_to_fix.copy(), validation_report, failure_feedback)

        if remediated_task_attempt.get('status') == 'FATAL_ERROR':
            print("   [ArchitectNode] <- Архитектор не смог внести исправления. Сигнализирую о фатальной ошибке.")
            state.setdefault('node_outputs', {})['architect_status'] = 'FATAL_ERROR'
            state['completed_tasks'].append(state.pop('current_task')) # Записываем исходную задачу как проваленную
            return state
        
        # Если инструмент создан, остаемся в цикле для следующей проверки
        print("   [ArchitectNode] Инструмент успешно создан. Повторная проверка выполнимости...")

    # Если вышли из цикла
    print(f"   [ArchitectNode] !!! Не удалось сделать задачу выполнимой после {max_iterations} итераций.")
    state.setdefault('node_outputs', {})['architect_status'] = 'FATAL_ERROR'
    task_to_fix = state.pop('current_task')
    task_to_fix['status'] = 'FATAL_ERROR'
    state['completed_tasks'].append(task_to_fix)
    return state



def outline_node(state: GraphState, outline_agent: OutlineAgent) -> GraphState:
    """Узел для создания плана финального отчета."""
    print("\n--- Узел: Report Outline ---")
    task = {
        "task_id": "outline_generation",
        "knowledge_base": state.get("knowledge_base", {})
    }
    outline = outline_agent.execute(task, "gemini-2.5-pro", state.get('user_config', {})) # Было: "gemini-2.5-flash"
    state['report_outline'] = outline
    state['drafted_sections'] = [] # Инициализируем список для черновиков
    return state

def section_fetcher_node(state: GraphState) -> GraphState:
    """Узел, который берет следующую секцию из плана для написания."""
    print("\n--- Узел: Section Fetcher ---")
    outline_sections = state.get('report_outline', {}).get('sections', [])
    num_drafted = len(state.get('drafted_sections', []))
    
    if num_drafted < len(outline_sections):
        section_to_draft = outline_sections[num_drafted]
        state['current_section_to_draft'] = section_to_draft
        print(f"   [SectionFetcher] -> Взял в работу секцию: '{section_to_draft.get('section_title')}'")
    else:
        state['current_section_to_draft'] = None
    return state

def section_writer_node(state: GraphState, section_writer_agent: SectionWriterAgent) -> GraphState:
    """Узел для написания текста одной секции."""
    print("\n--- Узел: Section Writer ---")
    section_to_draft = state.get('current_section_to_draft')
    if not section_to_draft:
        return state
        
    task = {
        "task_id": f"write_section_{section_to_draft.get('section_title', '').replace(' ', '_')}",
        "section_to_draft": section_to_draft,
        "knowledge_base": state.get("knowledge_base", {})
    }
    
    # Используем дешевую модель для написания черновиков
    written_section = section_writer_agent.execute(task, "gemini-2.5-flash", state.get('user_config', {})) # Было: "gemma-3"
    
    # Добавляем название секции к результату для компилятора
    full_section_data = {
        "section_title": section_to_draft.get('section_title'),
        "markdown_content": written_section.get('markdown_content', '')
    }
    state['drafted_sections'].append(full_section_data)
    return state

def final_compile_node(state: GraphState, writer: ReportWriterAgent, output_dir: str) -> GraphState:
    """Узел для финальной сборки отчета."""
    print("\n--- Узел: Final Compilation ---")
    task = {
        "task_id": "final_compilation",
        "drafted_sections": state.get('drafted_sections', []),
        "report_title": state.get('report_outline', {}).get('title', "Аналитический отчет")
    }
    final_markdown = writer.execute(task, "gemini-2.5-pro", state.get('user_config', {})) # Было: "gemini-2.5-flash"
    
    # Пост-обработка цитат
    final_markdown_with_citations = citation_post_processor(final_markdown, state.get('knowledge_base', {}))
    
    report_path = os.path.join(output_dir, "Final_Report_v4.2.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_markdown_with_citations)
    print(f"   [FinalCompileNode] -> Финальный отчет сохранен в {report_path}")
    return state


def revision_node(state: GraphState, reviser: ReviserAgent, supervisor: SupervisorAgent) -> GraphState:
    print("\n--- Узел: Revision ---")
    reviser_task = {"task_id": "revision_01", "description": "Провести ревизию текущего состояния исследования.", "knowledge_base": state.get("knowledge_base", {}), "remaining_tasks": state.get("task_queue", [])}
    revision_report = reviser.execute(reviser_task, "gemini-2.5-pro", state.get('user_config', {}))
    state.setdefault('node_outputs', {})['revision_report'] = revision_report
    if not revision_report.get('is_sufficient') and revision_report.get('new_task_suggestions'):
        print("   [RevisionNode] -> Ревизор счел информацию недостаточной. Запрашиваю у Supervisor'а новые задачи...")
        feedback_prompt = f"Критик дал следующий фидбек: '{revision_report['feedback']}'. Сгенерируй план из следующих задач: {revision_report['new_task_suggestions']}"
        new_plan = supervisor.create_initial_plan({"user_context": {"main_goal": feedback_prompt}})
        if new_plan and new_plan.get('tasks'):
            print(f"   [RevisionNode] <- Получено {len(new_plan['tasks'])} новых корректирующих задач.")
            state['task_queue'].extend(new_plan['tasks'])
            state['model_assignments'].update(new_plan.get('initial_model_assignments', {}))
    return state

# ====================================================================================
# === 2. ОПРЕДЕЛЕНИЕ МАРШРУТИЗАТОРОВ (CONDITIONAL EDGES) =============================
# ====================================================================================

def task_router(state: GraphState) -> str:
    """Маршрутизатор, который решает, есть ли задачи или нужно завершать работу."""
    print("\n--- Узел: Task Router ---")
    if state.get('task_queue'):
        return "fetcher"
    else:
        # Если задач нет, проверяем, есть ли план отчета. Если нет, создаем.
        if not state.get('report_outline'):
             print("   [TaskRouter] -> Очередь пуста. Перехожу к созданию плана отчета.")
             return "outline"
        else:
             print("   [TaskRouter] -> Все задачи и отчеты завершены. Конец работы.")
             return END

def post_execution_router(state: GraphState) -> str:
    """
    НОВЫЙ МАРШРУТИЗАТОР. Решает, что делать после выполнения задачи.
    """
    print("\n--- Узел: Post-Execution Router ---")
    last_task = state['completed_tasks'][-1]

    if last_task['status'] == 'FAILURE':
        print(f"   [PostExecRouter] !!! Задача {last_task['task_id']} провалена. Передаю на анализ.")
        return "failure_analyst"

    # Определяем тип задачи по имени агента (можно использовать и префиксы в task_id)
    agent_name = last_task['agent_name']
    is_research_task = agent_name in ["SingleStepToolAgent"]
    is_artifact_task = agent_name in ["FinancialModelAgent", "ProductManagerAgent", "RoadmapVisualizationAgent"]

    if is_research_task:
        print("   [PostExecRouter] -> Исследовательская задача. Отправляю на обработку в KB.")
        return "kb_ingestion"
    elif is_artifact_task:
        print("   [PostExecRouter] -> Задача генерации артефакта. Сохраняю артефакт.")
        return "artifact_commit"
    else:
        # Для всех остальных (аналитических, ревизионных и т.д.) просто берем следующую задачу
        print(f"   [PostExecRouter] -> Задача типа '{agent_name}'. Беру следующую задачу.")
        return "task_router"

    
def final_audit_router(state: GraphState) -> str:
    """
    Маршрутизатор, который решает, завершать ли граф на основе вердикта аудитора.
    """
    print("\n--- Узел: Final Audit Router ---")
    report = state.get('node_outputs', {}).get('final_audit_report', {})
    
    if report.get('is_complete'):
        print("   [FinalAuditRouter] -> Аудит пройден. Цель достигнута. Завершение работы.")
        return END
    else:
        new_tasks = report.get('new_tasks', [])
        if new_tasks:
            print(f"   [FinalAuditRouter] -> Аудит выявил недочеты. Добавляю {len(new_tasks)} новых задач.")
            state['task_queue'].extend(new_tasks)
            # Предполагаем, что в новых задачах есть model_assignments, если нет - нужно будет расширить
            for task in new_tasks:
                if 'model_name' in task:
                    state['model_assignments'][task['task_id']] = task['model_name']
            return "fetcher"
        else:
            print("   [FinalAuditRouter] -> Аудит не пройден, но новых задач не предложено. Завершение работы во избежание цикла.")
            return END

def post_execution_router(state: GraphState) -> str:
    """
    Маршрутизатор, который решает, что делать после выполнения задачи.
    """
    print("\n--- Узел: Post-Execution Router ---")
    last_completed_task = state['completed_tasks'][-1]

    if last_completed_task['status'] == 'SUCCESS':
        print("   [PostExecRouter] -> Задача успешна. Проверяю, что делать дальше.")
        state['escalation_count'] = 0
        is_research_task = not last_completed_task['task_id'].startswith('artifact_')
        remaining_research_tasks = any(not t['task_id'].startswith('artifact_') for t in state['task_queue'])
        
        if is_research_task and not remaining_research_tasks:
            print("   [PostExecRouter] -> Последняя исследовательская задача выполнена. Перехожу к ревизии.")
            return "revision"
        
        print("   [PostExecRouter] -> Исследование продолжается. Беру следующую задачу.")
        return "fetcher"
    else:
        print(f"   [PostExecRouter] !!! Задача {last_completed_task['task_id']} провалена. Передаю на анализ.")
        return "failure_analyst"

def failure_router(state: GraphState) -> str:
    """
    Маршрутизатор, который направляет граф на основе вердикта FailureAnalystAgent.
    """
    print("\n--- Узел: Failure Router ---")
    report = state.get('node_outputs', {}).get('failure_analysis_report', {})
    action = report.get('action', 'FATAL_ERROR')
    data = report.get('data', {})

    print(f"   [FailureRouter] -> Вердикт аналитика: '{action}'.")

    if action == 'RETRY':
        failed_task = state['completed_tasks'].pop()
        failed_task['status'] = 'PENDING'
        state['task_queue'].insert(0, failed_task)
        return "fetcher"

    if action == 'RETRY_WITH_NEW_MODEL':
        next_model = data.get('next_model_name')
        if not next_model:
             print("   [FailureRouter] !!! Аналитик предложил сменить модель, но не указал какую. Фатальная ошибка.")
             return END
        
        failed_task = state['completed_tasks'].pop()
        failed_task['status'] = 'PENDING'
        state['model_assignments'][failed_task['task_id']] = next_model
        state['task_queue'].insert(0, failed_task)
        print(f"   [FailureRouter] Эскалирую задачу на модель '{next_model}'.")
        return "fetcher"

    if action == 'REGENERATE_TOOL':
        state.setdefault('node_outputs', {})['architect_feedback'] = data.get('feedback')
        return "prepare_for_architect"

    if action == 'CREATE_NEW_TASK':
        new_desc = data.get('new_task_description')
        if not new_desc:
            print("   [FailureRouter] !!! Аналитик предложил создать новую задачу, но не дал описание. Фатальная ошибка.")
            return END
        
        last_failed_task = state['completed_tasks'][-1]
        new_task = last_failed_task.copy()
        new_task['task_id'] = f"{last_failed_task['task_id']}_remediated"
        new_task['description'] = new_desc
        new_task['status'] = 'PENDING'
        state['task_queue'].insert(0, new_task)
        print("   [FailureRouter] Создана новая, скорректированная задача.")
        return "fetcher"

    print("   [FailureRouter] !!! Неустранимая ошибка. Аварийное завершение.")
    return END

def validation_router(state: GraphState) -> str:
    """
    Направляет поток выполнения на основе отчета от ValidatorAgent.
    """
    print("\n--- Узел: Validation Router ---")
    report = state.get('node_outputs', {}).get('validation_report', {})
    if report.get('is_executable'):
        print("   [ValidationRouter] -> Задача выполнима. Передаю исполнителю.")
        return "executor"
    else:
        print("   [ValidationRouter] -> Недостаточно инструментов. Передаю архитектору.")
        return "architect"

# --- НОВЫЙ МАРШРУТИЗАТОР ДЛЯ "АВАРИЙНОГО ТОРМОЗА" ---
def architect_router(state: GraphState) -> str:
    """
    Проверяет результат работы Архитектора.
    """
    print("\n--- Узел: Architect Router ---")
    status = state.get('node_outputs', {}).get('architect_status', 'FATAL_ERROR')
    if status == 'REMEDIATED':
        print("   [ArchitectRouter] -> Задача исправлена. Возвращаюсь к исполнению.")
        return "fetcher"
    else:
        print("   [ArchitectRouter] -> Архитектор не смог исправить задачу. АВАРИЙНОЕ ЗАВЕРШЕНИЕ.")
        return END

def qa_router(state: GraphState) -> str:
    """Маршрутизатор для конвейера QA."""
    print("\n--- Узел: QA Router ---")
    if state['node_outputs'].get('fixable'):
        return "fixer"
    else:
        state['node_outputs']['candidates_for_sanity_check'] = state['node_outputs'].get('good', [])
        return "sanity_check"

def reassessment_router(state: GraphState) -> str:
    """Маршрутизатор после повторной оценки исправленных фактов."""
    print("\n--- Узел: Re-assessment Router ---")
    good_initial = state['node_outputs'].get('good_initial', [])
    fixed_reassessed_good = state['node_outputs'].get('good', [])
    state['node_outputs']['candidates_for_sanity_check'] = good_initial + fixed_reassessed_good
    return "sanity_check"

def reflection_router(state: GraphState) -> str:
    """Маршрутизатор после рефлексии."""
    print("\n--- Узел: Reflection Router ---")
    if state['task_queue']:
        return "fetcher"
    else:
        return "final_report"
    
    
def revision_router(state: GraphState) -> str:
    """
    Маршрутизатор, который решает, продолжать ли исследование или переходить к QA.
    """
    print("\n--- Узел: Revision Router ---")
    revision_report = state.get('node_outputs', {}).get('revision_report', {})
    
    if not revision_report.get('is_sufficient'):
        print("   [RevisionRouter] -> Добавлены новые задачи. Возвращаюсь к исполнителю.")
        return "fetcher"
    else:
        print("   [RevisionRouter] -> Информация признана достаточной. Перехожу к QA.")
        # Подготавливаем данные для QA, как это делалось раньше
        state['node_outputs']['facts_for_assessment'] = state['node_outputs'].get('accumulated_raw_facts', [])
        return "qa"
    
def section_writing_router(state: GraphState) -> str:
    """Маршрутизатор, управляющий циклом написания секций."""
    print("\n--- Узел: Section Writing Router ---")
    outline_sections = state.get('report_outline', {}).get('sections', [])
    num_drafted = len(state.get('drafted_sections', []))
    
    if num_drafted < len(outline_sections):
        print(f"   [SectionRouter] -> Написано {num_drafted}/{len(outline_sections)}. Продолжаю цикл.")
        return "fetcher"
    else:
        print("   [SectionRouter] -> Все секции написаны. Перехожу к финальной компиляции.")
        return "compiler"


# ====================================================================================
# === 3. СБОРКА ГРАФА (WORKFLOW COMPILATION) =========================================
# ====================================================================================

def build_graph(agents: dict, output_dir: str):
    workflow = StateGraph(GraphState)

    # --- Регистрация узлов ---
    workflow.add_node("supervisor", lambda state: supervisor_node(state, agents['Supervisor']))
    workflow.add_node("task_fetcher", task_fetcher_node)
    workflow.add_node("task_executor", lambda state: task_executor_node(state, agents))
    workflow.add_node("kb_ingestion", kb_ingestion_node)
    workflow.add_node("artifact_commit", artifact_commit_node)
    workflow.add_node("commit_to_kb", commit_to_kb_node)

    # Узлы QA
    workflow.add_node("qa", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("fixer", lambda state: fixer_node(state, agents['Fixer']))
    workflow.add_node("sanity_check", lambda state: sanity_check_node(state, agents['SanityCheckCritic']))

    # Узлы обработки сбоев
    workflow.add_node("failure_analyst", lambda state: failure_analyst_node(state, agents['FailureAnalyst']))
    workflow.add_node("architect", lambda state: architect_node(state, agents['Architect'], agents['Validator']))
    workflow.add_node("tool_validator", lambda state: tool_validator_node(state, agents['Validator']))

    # Узлы генерации отчета
    workflow.add_node("outline", lambda state: outline_node(state, agents['OutlineAgent']))
    workflow.add_node("section_writer", lambda state: section_writer_node(state, agents['SectionWriterAgent']))
    workflow.add_node("final_compiler", lambda state: final_compile_node(state, agents['ReportWriter'], output_dir))

    # --- Определение потока управления (ребер графа) ---
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "task_fetcher")

    # 1. Основной цикл: проверка наличия задач
    workflow.add_conditional_edges(
        "task_fetcher",
        lambda s: "validator" if s.get("current_task") else "task_router",
        {"validator": "tool_validator", "task_router": "task_router"}
    )

    # 2. Валидация и выполнение задачи
    workflow.add_conditional_edges("tool_validator", validation_router, {
        "executor": "task_executor",
        "architect": "architect"
    })
    workflow.add_edge("task_executor", "post_execution_router")

    # 3. Маршрутизация после выполнения
    workflow.add_conditional_edges(
        "post_execution_router",
        lambda s: s['completed_tasks'][-1]['status'] if s.get('completed_tasks') else END,
        {
            "SUCCESS": "post_success_router", # Вложенный маршрутизатор для успеха
            "FAILURE": "failure_analyst"
        }
    )
    
    # 3a. Вложенный маршрутизатор для успешных задач
    workflow.add_conditional_edges(
        "post_success_router",
        lambda s: agents[s['completed_tasks'][-1]['agent_name']].__class__.__name__, # Определяем путь по имени агента
        {
            "SingleStepToolAgent": "kb_ingestion",
            "FinancialModelAgent": "artifact_commit",
            "ProductManagerAgent": "artifact_commit",
            # ... другие агенты-артефакторы
            "default": "task_router" # Для всех остальных - к следующей задаче
        }
    )

    # 4. Ветки обработки результатов
    workflow.add_edge("kb_ingestion", "qa")
    workflow.add_edge("artifact_commit", "task_router")

    # 5. Конвейер QA
    workflow.add_conditional_edges("qa", qa_router, {
        "fixer": "fixer",
        "sanity_check": "sanity_check"
    })
    workflow.add_edge("fixer", "qa") # После исправления - снова на оценку
    workflow.add_edge("sanity_check", "commit_to_kb")
    workflow.add_edge("commit_to_kb", "task_router") # После коммита - к следующей задаче

    # 6. Ветка обработки сбоев (без изменений)
    workflow.add_conditional_edges("failure_analyst", failure_router, {
        "fetcher": "task_fetcher",
        "architect": "architect",
        END: END
    })
    workflow.add_conditional_edges("architect", architect_router, {
        "fetcher": "task_fetcher",
        END: END
    })

    # 7. Ветка генерации отчета (запускается из task_router, когда задачи кончились)
    workflow.add_edge("outline", "task_router") # После создания плана - снова в роутер, который найдет задачи на написание секций
    workflow.add_edge("section_writer", "task_router")
    workflow.add_edge("final_compiler", END) # Конец

    # 8. Финальный маршрутизатор
    workflow.add_conditional_edges("task_router", task_router, {
        "fetcher": "task_fetcher",
        "outline": "outline",
        END: END
    })

    return workflow.compile()



# ====================================================================================
# === 4. ФУНКЦИЯ ЗАПУСКА ГРАФА =======================================================
# ====================================================================================

def run(app, initial_state: GraphState, state_file_path: str):
    try:
        # Увеличиваем лимит рекурсии для сложных графов
        config = {"recursion_limit": 100}
        for event in app.stream(initial_state, config, stream_mode="values"):
            try:
                with open(state_file_path, "w", encoding="utf-8") as f:
                    # Используем default=str для сериализации объектов, которые не являются JSON-сериализуемыми
                    json.dump(event, f, ensure_ascii=False, indent=2, default=str)
            except (IOError, TypeError) as e:
                print(f"!!! [Orchestrator] ВНИМАНИЕ: Не удалось сохранить состояние. Ошибка: {e}")
        print("\n--- ВЫПОЛНЕНИЕ ГРАФА ЗАВЕРШЕНО ---")
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
            print(f"   [Orchestrator] Файл состояния '{state_file_path}' удален после успешного завершения.")
    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ ГРАФА: {e}")
        traceback.print_exc()
        print(f"   [Orchestrator] Промежуточное состояние сохранено в '{state_file_path}' для возобновления.")
