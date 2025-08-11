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
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ValidatorAgent
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
    print(f"   [FetcherNode] -> Взял в работу задачу: {task['task_id']} ({task['agent_name']})")
    return state

def tool_validator_node(state: GraphState, validator: ValidatorAgent) -> GraphState:
    """
    Проверяет, можно ли выполнить задачу с текущими инструментами, ПЕРЕД ее запуском.
    """
    print("\n--- Узел: Tool Validator ---")
    task = state.get('current_task')
    if not task: return state

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
            result = agent.execute(task, model, state.copy())
            
            print("\n" + "-"*25 + " НАЧАЛО ОТВЕТА АГЕНТА " + "-"*25)
            print(f"Сырой результат от агента '{agent_name}':")
            print(result)
            print("-" * 25 + " КОНЕЦ ОТВЕТА АГЕНТА " + "-"*27 + "\n")

            if not result:
                task['status'] = 'FAILURE'
                state['error_message'] = "Агент не вернул результат. Вероятная причина - ошибка API или внутренняя ошибка агента. См. лог выше."
            else:
                task['status'] = 'SUCCESS'
                if agent_name in ["Researcher", "Contrarian"]:
                    state['node_outputs']['accumulated_raw_facts'].extend(result)
                elif agent_name in ["FinancialModelAgent", "ProductManagerAgent"]:
                    state.setdefault('artifacts', {})[task['task_id']] = result

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

def reflection_node(state: GraphState, analyst: AnalystAgent, supervisor: SupervisorAgent) -> GraphState:
    print("\n--- Узел: Reflection ---")
    user_config = state.get('user_config', {})
    analysis_result = analyst.execute_reflection(state['knowledge_base'], "gemini-2.5-flash", user_config)
    if not analysis_result or not analysis_result.get('data'):
        state['task_queue'] = []
        return state
    next_phase_plan = supervisor.create_next_phase_plan(analysis_result['data'])
    new_tasks = next_phase_plan.get('tasks', [])
    if new_tasks:
        state['task_queue'].extend(new_tasks)
        state['model_assignments'].update(next_phase_plan.get('initial_model_assignments', {}))
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

def architect_node(state: GraphState, architect: ArchitectAgent) -> GraphState:
    """
    Узел для самокоррекции. Работает по новой, проактивной схеме.
    """
    print("\n--- Узел: Architect ---")
    
    task_to_fix = state.get('current_task')
    validation_report = state.get('node_outputs', {}).get('validation_report')

    # === ИЗМЕНЕНИЕ НАЧАТО: Убрана хрупкая проверка на validation_report ===
    if not task_to_fix:
        print("!!! [ArchitectNode] КРИТИЧЕСКАЯ ОШИБКА: Нет текущей задачи для работы. Архитектор не может продолжить.")
        state.setdefault('node_outputs', {})['architect_status'] = 'FATAL_ERROR'
        return state
    # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    print(f"   [ArchitectNode] Взял на исправление задачу: {task_to_fix.get('task_id')}")
    
    # Теперь агент сам должен быть устойчив к отсутствию отчета
    remediated_task = architect.fix_or_enhance(task_to_fix, validation_report)
    
    if remediated_task.get('status') == 'FATAL_ERROR':
        print("   [ArchitectNode] <- Архитектор не смог внести исправления. Сигнализирую о фатальной ошибке.")
        state.setdefault('node_outputs', {})['architect_status'] = 'FATAL_ERROR'
        state['completed_tasks'].append(remediated_task)
    else:
        print(f"   [ArchitectNode] <- Задача {task_to_fix['task_id']} исправлена и возвращена в очередь.")
        state['task_queue'].insert(0, remediated_task)
        state.setdefault('node_outputs', {})['architect_status'] = 'REMEDIATED'

    state['escalation_count'] = 0
    state['current_task'] = None
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

def escalation_router(state: GraphState) -> str:
    """
    Маршрутизатор, реализующий логику Каскадной Эскалации.
    """
    print("\n--- Узел: Escalation Router ---")
    if not state['completed_tasks']: return "fetcher"
    last_completed_task = state['completed_tasks'][-1]
    
    if last_completed_task['status'] == 'SUCCESS':
        state['escalation_count'] = 0
        is_research = not last_completed_task['task_id'].startswith('artifact_')
        remaining_research = any(not t['task_id'].startswith('artifact_') for t in state['task_queue'])
        if is_research and not remaining_research: return "revision"
        return "fetcher"

    print(f"   [EscalationRouter] !!! Задача {last_completed_task['task_id']} провалена.")
    
    escalation_count = state.get('escalation_count', 0) + 1
    state['escalation_count'] = escalation_count
    
    # Проверяем лимит эскалаций. MAX_ESCALATIONS_PER_TASK = 2 означает, что после 3-го сбоя (1-й + 2 эскалации) идем к архитектору.
    if escalation_count > MAX_ESCALATIONS_PER_TASK:
        print(f"   [EscalationRouter] !!! Лимит эскалаций ({MAX_ESCALATIONS_PER_TASK}) для задачи исчерпан. Передаю Архитектору.")
        # === ИЗМЕНЕНИЕ НАЧАТО: Маршрут изменен на узел-подготовитель ===
        return "prepare_for_architect"
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    current_model = state['model_assignments'][last_completed_task['task_id']]
    next_model = MODEL_ESCALATION_PATH.get(current_model)
    
    if not next_model:
        print(f"   [EscalationRouter] !!! Модель '{current_model}' на пределе эскалации. Передаю Архитектору.")
        # === ИЗМЕНЕНИЕ НАЧАТО: Маршрут изменен на узел-подготовитель ===
        return "prepare_for_architect"
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    print(f"   [EscalationRouter] -> Эскалирую задачу на модель '{next_model}' (Попытка {escalation_count} из {MAX_ESCALATIONS_PER_TASK}).")
    state['model_assignments'][last_completed_task['task_id']] = next_model
    failed_task = state['completed_tasks'].pop()
    failed_task['status'] = 'PENDING'
    state['task_queue'].insert(0, failed_task)
    return "fetcher"

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
    
def artifact_router(state: GraphState) -> str:
    """
    Маршрутизатор, который проверяет, есть ли в очереди задачи на создание артефактов.
    """
    print("\n--- Узел: Artifact Router ---")
    # Ищем в оставшейся очереди задачи с префиксом 'artifact_'
    has_artifact_tasks = any(task['task_id'].startswith('artifact_') for task in state['task_queue'])
    
    if has_artifact_tasks:
        print("   [ArtifactRouter] -> Обнаружены задачи на создание артефактов. Возвращаюсь к исполнителю.")
        return "fetcher"
    else:
        print("   [ArtifactRouter] -> Задачи на создание артефактов отсутствуют. Перехожу к очистке.")
        return "janitor"
    
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

    workflow.add_node("supervisor", lambda state: supervisor_node(state, agents['Supervisor']))
    workflow.add_node("task_fetcher", task_fetcher_node)
    workflow.add_node("tool_validator", lambda state: tool_validator_node(state, agents['Validator']))
    workflow.add_node("task_executor", lambda state: task_executor_node(state, agents))
    workflow.add_node("prepare_for_architect", prepare_for_architect_node)
    workflow.add_node("architect", lambda state: architect_node(state, agents['Architect']))
    workflow.add_node("revision", lambda state: revision_node(state, agents['Reviser'], agents['Supervisor']))
    workflow.add_node("qa", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("fixer", lambda state: fixer_node(state, agents['Fixer']))
    workflow.add_node("qa_reassessment", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("sanity_check", lambda state: sanity_check_node(state, agents['SanityCheckCritic']))
    workflow.add_node("commit", commit_node)
    workflow.add_node("janitor", lambda state: janitor_node(state, agents['Janitor']))
    workflow.add_node("reflection", lambda state: reflection_node(state, agents['Analyst'], agents['Supervisor']))
    workflow.add_node("outline", lambda state: outline_node(state, agents['OutlineAgent']))
    workflow.add_node("section_fetcher", section_fetcher_node)
    workflow.add_node("section_writer", lambda state: section_writer_node(state, agents['SectionWriterAgent']))
    workflow.add_node("final_compiler", lambda state: final_compile_node(state, agents['ReportWriter'], output_dir))

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "task_fetcher")
    
    workflow.add_edge("task_fetcher", "tool_validator")
    workflow.add_conditional_edges("tool_validator", validation_router, {
        "executor": "task_executor",
        "architect": "architect",
        END: END
    })
    
    # === ИЗМЕНЕНИЕ НАЧАТО: Восстановлен путь к архитектору ===
    workflow.add_conditional_edges("task_executor", escalation_router, {
        "fetcher": "task_fetcher", 
        "revision": "revision",
        "prepare_for_architect": "prepare_for_architect", # Новый маршрут
        END: END
    })
    
    # Новый маршрут для самокоррекции
    workflow.add_edge("prepare_for_architect", "architect")
    # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===
    
    workflow.add_conditional_edges("architect", architect_router, {
        "fetcher": "task_fetcher",
        END: END
    })

    workflow.add_conditional_edges("revision", revision_router, {"fetcher": "task_fetcher", "qa": "qa"})
    workflow.add_conditional_edges("qa", qa_router, {"fixer": "fixer", "sanity_check": "sanity_check"})
    workflow.add_edge("fixer", "qa_reassessment")
    workflow.add_conditional_edges("qa_reassessment", reassessment_router, {"sanity_check": "sanity_check"})
    workflow.add_edge("sanity_check", "commit")
    workflow.add_conditional_edges("commit", artifact_router, {"fetcher": "task_fetcher", "janitor": "janitor"})
    workflow.add_edge("janitor", "reflection")
    workflow.add_conditional_edges("reflection", reflection_router, {"fetcher": "task_fetcher", "final_report": "outline"})
    workflow.add_edge("outline", "section_fetcher")
    workflow.add_edge("section_fetcher", "section_writer")
    workflow.add_conditional_edges("section_writer", section_writing_router, {"fetcher": "section_fetcher", "compiler": "final_compiler"})
    workflow.add_edge("final_compiler", END)

    return workflow.compile()


# ====================================================================================
# === 4. ФУНКЦИЯ ЗАПУСКА ГРАФА =======================================================
# ====================================================================================

def run(app, initial_state: GraphState, state_file_path: str):
    try:
        for event in app.stream(initial_state, stream_mode="values"):
            try:
                with open(state_file_path, "w", encoding="utf-8") as f:
                    json.dump(event, f, ensure_ascii=False, indent=2)
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