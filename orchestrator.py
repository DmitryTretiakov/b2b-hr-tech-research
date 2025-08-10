# orchestrator.py
from langgraph.graph import StateGraph, END
from core.state import GraphState
from agents.supervisor import SupervisorAgent
from agents.workers import (
    ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, 
    AnalystAgent, ReportWriterAgent, SanityCheckCritic,
    FinancialModelAgent, ProductManagerAgent
)
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent
from utils.helpers import citation_post_processor
import os
import json

# --- Константы для эскалации ---
MAX_ESCALATIONS = 1
MODEL_ESCALATION_PATH = {
    "gemma-3": "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
    "gemini-2.5-flash": "gemini-2.5-flash" # Предел эскалации
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
    Узел, который берет следующую задачу из очереди и помещает ее в 'current_task'.
    """
    print("\n--- Узел: Task Fetcher ---")
    if not state['task_queue']:
        return state
    
    task = state['task_queue'].pop(0)
    state['current_task'] = task
    state['error_message'] = None
    print(f"   [FetcherNode] -> Взял в работу задачу: {task['task_id']} ({task['agent_name']})")
    return state

def task_executor_node(state: GraphState, agents: dict) -> GraphState:
    """
    Выполняет одну задачу. Для агентов-артефакторов внедряет Базу Знаний в задачу.
    """
    print("\n--- Узел: Task Executor ---")
    task = state.get('current_task')
    if not task: return state
    
    agent_name = task['agent_name']
    agent = agents.get(agent_name)
    model = state['model_assignments'].get(task['task_id'])
    user_config = state.get('user_config', {})

    if not agent or not model:
        task['status'] = 'FAILURE'
        state['error_message'] = f"Агент {agent_name} или модель не найдены."
        state['completed_tasks'].append(task)
        state['current_task'] = None
        return state
    
    # --- ИНЪЕКЦИЯ КОНТЕКСТА ДЛЯ АГЕНТОВ-АРТЕФАКТОРОВ ---
    if agent_name in ["FinancialModelAgent", "ProductManagerAgent"]:
        print(f"   [Executor] Внедряю полную Базу Знаний в задачу для {agent_name}...")
        task["knowledge_base"] = state.get("knowledge_base", {})

    try:
        result = agent.execute(task, model, user_config)
        task['status'] = 'SUCCESS'
        
        if agent_name in ["Researcher", "Contrarian"]:
            state['node_outputs']['accumulated_raw_facts'].extend(result)
        # --- СОХРАНЕНИЕ АРТЕФАКТОВ ---
        elif agent_name in ["FinancialModelAgent", "ProductManagerAgent"]:
            state.setdefault('artifacts', {})[task['task_id']] = result
            print(f"   [Executor] <- Артефакт '{task['task_id']}' успешно создан и сохранен.")

    except Exception as e:
        task['status'] = 'FAILURE'
        state['error_message'] = str(e)
    
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
    model = "gemini-2.5-flash"
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

def architect_node(state: GraphState, architect: ArchitectAgent) -> GraphState:
    print("\n--- Узел: Architect ---")
    task = state['current_task']
    error = state.get('error_message', 'Нет деталей')
    remediated_task = architect.fix_or_enhance(task, error)
    state['task_queue'].insert(0, remediated_task)
    state['escalation_count'] = 0
    state['current_task'] = None
    return state

def final_report_node(state: GraphState, analyst: AnalystAgent, writer: ReportWriterAgent, output_dir: str) -> GraphState:
    print("\n--- Узел: Final Report ---")
    user_config = state.get('user_config', {})
    analysis_data = analyst.execute_final_synthesis(state['knowledge_base'], "gemini-2.5-flash", user_config)
    if not analysis_data: return state
    report_content = writer.execute(analysis_data, "gemma-3", user_config)
    if not report_content: return state
    final_markdown = citation_post_processor(report_content, state['knowledge_base'])
    report_path = os.path.join(output_dir, "Final_Report_v4.1.md")
    with open(report_path, "w", encoding="utf-8") as f: f.write(final_markdown)
    return state

# ====================================================================================
# === 2. ОПРЕДЕЛЕНИЕ МАРШРУТИЗАТОРОВ (CONDITIONAL EDGES) =============================
# ====================================================================================

# ... (все маршрутизаторы остаются без изменений) ...
def escalation_router(state: GraphState) -> str:
    """Маршрутизатор, реализующий логику Каскадной Эскалации."""
    print("\n--- Узел: Escalation Router ---")
    last_completed_task = state['completed_tasks'][-1]
    task_id = last_completed_task['task_id']

    if last_completed_task['status'] == 'SUCCESS':
        print(f"   [EscalationRouter] Задача {task_id} успешна.")
        state['escalation_count'] = 0
        if state['task_queue']:
            return "fetcher"
        else:
            print("   [EscalationRouter] -> Фаза исследования завершена. Перехожу к QA.")
            state['node_outputs']['facts_for_assessment'] = state['node_outputs'].get('accumulated_raw_facts', [])
            return "qa"

    print(f"   [EscalationRouter] !!! Задача {task_id} провалена.")
    escalation_count = state.get('escalation_count', 0)
    
    if escalation_count >= MAX_ESCALATIONS:
        print(f"   [EscalationRouter] !!! Лимит эскалаций ({MAX_ESCALATIONS}) исчерпан. Передаю Архитектору.")
        state['current_task'] = state['completed_tasks'].pop()
        return "architect"

    current_model = state['model_assignments'][task_id]
    next_model = MODEL_ESCALATION_PATH.get(current_model)

    if not next_model or next_model == current_model:
        print(f"   [EscalationRouter] !!! Модель '{current_model}' на пределе эскалации. Передаю Архитектору.")
        state['current_task'] = state['completed_tasks'].pop()
        return "architect"

    print(f"   [EscalationRouter] -> Эскалирую задачу {task_id} на модель '{next_model}'.")
    state['escalation_count'] = escalation_count + 1
    state['model_assignments'][task_id] = next_model
    
    failed_task = state['completed_tasks'].pop()
    failed_task['status'] = 'PENDING'
    state['task_queue'].insert(0, failed_task)
    
    return "fetcher"

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

# ====================================================================================
# === 3. СБОРКА ГРАФА (WORKFLOW COMPILATION) =========================================
# ====================================================================================

def build_graph(agents: dict, output_dir: str):
    """Собирает и компилирует финальный граф LangGraph."""
    workflow = StateGraph(GraphState)

    # Добавляем все узлы, включая новые
    workflow.add_node("supervisor", lambda state: supervisor_node(state, agents['Supervisor']))
    workflow.add_node("task_fetcher", task_fetcher_node)
    workflow.add_node("task_executor", lambda state: task_executor_node(state, agents))
    workflow.add_node("architect", lambda state: architect_node(state, agents['Architect']))
    workflow.add_node("qa", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("fixer", lambda state: fixer_node(state, agents['Fixer']))
    workflow.add_node("qa_reassessment", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("sanity_check", lambda state: sanity_check_node(state, agents['SanityCheckCritic']))
    workflow.add_node("commit", commit_node)
    workflow.add_node("janitor", lambda state: janitor_node(state, agents['Janitor']))
    workflow.add_node("reflection", lambda state: reflection_node(state, agents['Analyst'], agents['Supervisor']))
    workflow.add_node("final_report", lambda state: final_report_node(state, agents['Analyst'], agents['ReportWriter'], output_dir))

    # --- ПЕРЕСТРОЙКА ЛОГИКИ ГРАФА ---
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "task_fetcher")
    workflow.add_edge("task_fetcher", "task_executor")
    
    # Основной цикл исследования и самокоррекции
    workflow.add_conditional_edges("task_executor", escalation_router, {
        "fetcher": "task_fetcher",
        "qa": "qa",
        "architect": "architect"
    })
    workflow.add_edge("architect", "task_fetcher")

    # Конвейер QA
    workflow.add_conditional_edges("qa", qa_router, {"fixer": "fixer", "sanity_check": "sanity_check"})
    workflow.add_edge("fixer", "qa_reassessment")
    workflow.add_conditional_edges("qa_reassessment", reassessment_router, {"sanity_check": "sanity_check"})
    workflow.add_edge("sanity_check", "commit")
    
    # --- НОВЫЙ УЗЕЛ В ЛОГИКЕ: ПОСЛЕ КОММИТА РЕШАЕМ, СОЗДАВАТЬ ЛИ АРТЕФАКТЫ ---
    workflow.add_conditional_edges("commit", artifact_router, {
        "fetcher": "task_fetcher", # Если есть задачи на артефакты, возвращаемся в цикл
        "janitor": "janitor"       # Если нет, идем дальше по старой ветке
    })
    
    # Цикл завершения фазы и рефлексии
    workflow.add_edge("janitor", "reflection")
    workflow.add_conditional_edges("reflection", reflection_router, {
        "fetcher": "task_fetcher",
        "final_report": "final_report"
    })

    workflow.add_edge("final_report", END)

    return workflow.compile()

# ====================================================================================
# === 4. ФУНКЦИЯ ЗАПУСКА ГРАФА =======================================================
# ====================================================================================

def run(app, initial_state: GraphState, state_file_path: str):
    """
    Запускает выполнение скомпилированного графа, сохраняя состояние после каждого шага.
    """
    try:
        for event in app.stream(initial_state, stream_mode="values"):
            # `event` содержит полное, актуальное состояние графа после каждого шага.
            # Сохраняем это состояние в файл.
            try:
                with open(state_file_path, "w", encoding="utf-8") as f:
                    json.dump(event, f, ensure_ascii=False, indent=2)
            except (IOError, TypeError) as e:
                print(f"!!! [Orchestrator] ВНИМАНИЕ: Не удалось сохранить состояние. Ошибка: {e}")

        print("\n--- ВЫПОЛНЕНИЕ ГРАФА ЗАВЕРШЕНО ---")
        # После успешного завершения можно удалить файл состояния, чтобы следующий запуск был чистым
        if os.path.exists(state_file_path):
            os.remove(state_file_path)
            print(f"   [Orchestrator] Файл состояния '{state_file_path}' удален после успешного завершения.")

    except Exception as e:
        print(f"\n!!! КРИТИЧЕСКАЯ ОШИБКА ВО ВРЕМЯ ВЫПОЛНЕНИЯ ГРАФА: {e}")
        import traceback
        traceback.print_exc()
        print(f"   [Orchestrator] Промежуточное состояние сохранено в '{state_file_path}' для возобновления.")