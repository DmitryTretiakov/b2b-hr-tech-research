# orchestrator.py
from langgraph.graph import StateGraph, END
from core.state import GraphState
from agents.supervisor import SupervisorAgent
from agents.workers import ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, AnalystAgent, ReportWriterAgent
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent

# Константы
MAX_ESCALATIONS = 1
MODEL_ESCALATION_PATH = {"gemma-3": "gemini-2.5-flash", "gemini-2.5-flash-lite": "gemini-2.5-flash"}

# --- Узлы Графа ---

def supervisor_node(state: GraphState, supervisor: SupervisorAgent) -> GraphState:
    print("--- Узел: Supervisor ---")
    if not state.get('task_queue'):
        plan = supervisor.create_initial_plan("Подготовить бизнес-кейс для HR-Tech продукта")
        state['task_queue'].extend(plan.get('tasks', []))
        state['model_assignments'].update(plan.get('initial_model_assignments', {}))
    return state

def worker_node(state: GraphState, agents: dict) -> GraphState:
    print("--- Узел: Worker ---")
    task = state['task_queue'].pop(0)
    state['current_task'] = task
    state['escalation_count'] = 0
    
    agent = agents.get(task.get('agent_name'))
    model_name = state['model_assignments'].get(task['task_id'])
    
    try:
        # ЗАГЛУШКА: пока все агенты вызываются одинаково, в будущем здесь будет разная логика
        result = agent.execute(task, model_name)
        state['current_task_result'] = result
    except Exception as e:
        state['current_task_result'] = {"status": "FAILURE", "error": str(e)}
    return state

def janitor_node(state: GraphState, janitor: KnowledgeJanitorAgent) -> GraphState:
    print("--- Узел: Janitor ---")
    state['knowledge_base'] = janitor.cleanup_knowledge_base(state['knowledge_base'])
    return state

def reflection_node(state: GraphState, analyst: AnalystAgent, supervisor: SupervisorAgent) -> GraphState:
    print("--- Узел: Reflection ---")
    analysis = analyst.execute(state['knowledge_base'], "gemini-2.5-flash")
    # ЗАГЛУШКА: здесь будет вызов supervisor.create_next_phase_plan(analysis)
    print("   [Reflection] Генерация следующей фазы (логика-заглушка).")
    return state

def architect_node(state: GraphState, architect: ArchitectAgent) -> GraphState:
    print("--- Узел: Architect ---")
    task = state['current_task']
    error = state.get('error_message', 'Нет деталей')
    fixed_task = architect.fix_task(task, error)
    state['task_queue'].insert(0, fixed_task) # Возвращаем исправленную задачу в начало очереди
    return state

def final_report_node(state: GraphState, analyst: AnalystAgent, writer: ReportWriterAgent) -> GraphState:
    print("--- Узел: Final Report ---")
    analysis = analyst.execute(state['knowledge_base'], "gemini-2.5-flash")
    report = writer.execute(analysis['data'], "gemma-3")
    print("\n\n==== ФИНАЛЬНЫЙ ОТЧЕТ ====\n", report['data'])
    return state

# --- Маршрутизатор ---

def router(state: GraphState) -> str:
    print("--- Узел: Router ---")
    result = state.get('current_task_result')
    
    if result and result.get('status') == 'FAILURE':
        if state['escalation_count'] < MAX_ESCALATIONS:
            state['escalation_count'] += 1
            task_id = state['current_task']['task_id']
            current_model = state['model_assignments'][task_id]
            next_model = MODEL_ESCALATION_PATH.get(current_model)
            if next_model:
                print(f"   [Router] -> ЭСКАЛАЦИЯ! Повтор задачи {task_id} на модели {next_model}.")
                state['model_assignments'][task_id] = next_model
                state['task_queue'].insert(0, state['current_task'])
                return "worker"
        print(f"   [Router] -> Предел эскалаций. Передаю управление Architect.")
        return "architect"

    if state['task_queue']:
        return "worker"
    
    # ЗАГЛУШКА: Пока нет логики фаз, после опустошения очереди переходим к отчету
    return "final_report"

# --- Сборка Графа ---

def build_graph(agents: dict):
    workflow = StateGraph(GraphState)

    workflow.add_node("supervisor", lambda state: supervisor_node(state, agents['Supervisor']))
    workflow.add_node("worker", lambda state: worker_node(state, agents))
    workflow.add_node("architect", lambda state: architect_node(state, agents['Architect']))
    workflow.add_node("final_report", lambda state: final_report_node(state, agents['Analyst'], agents['ReportWriter']))

    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "worker")
    workflow.add_conditional_edges("worker", router, {"worker": "worker", "architect": "architect", "final_report": "final_report"})
    workflow.add_edge("architect", "worker")
    workflow.add_edge("final_report", END)

    return workflow.compile()