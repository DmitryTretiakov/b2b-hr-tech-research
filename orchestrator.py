# orchestrator.py
from langgraph.graph import StateGraph, END
from core.state import GraphState
from agents.supervisor import SupervisorAgent
from agents.workers import ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, AnalystAgent, ReportWriterAgent, SanityCheckCritic
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent
from utils.helpers import citation_post_processor
import os

# --- Константы для эскалации ---
MAX_ESCALATIONS = 1
MODEL_ESCALATION_PATH = {
    "gemma-3": "gemini-2.5-flash",
    "gemini-2.5-flash-lite": "gemini-2.5-flash",
    "gemini-2.5-flash": "gemini-2.5-flash" # Предел эскалации
}

# ====================================================================================
# === 1. ОПРЕДЕЛЕНИЕ УЗЛОВ ГРАФА (NODES) =============================================
# ====================================================================================

def supervisor_node(state: GraphState, supervisor: SupervisorAgent) -> GraphState:
    """Узел для генерации первоначального плана, если его еще нет."""
    print("\n--- Узел: Supervisor ---")
    if not state.get('task_queue') and not state.get('completed_tasks'):
        print("   [SupervisorNode] Генерирую первоначальный план...")
        plan = supervisor.create_initial_plan("Подготовить бизнес-кейс для HR-Tech продукта")
        state['task_queue'].extend(plan.get('tasks', []))
        state['model_assignments'].update(plan.get('initial_model_assignments', {}))
    else:
        print("   [SupervisorNode] План уже существует, пропускаю генерацию.")
    return state

def research_node(state: GraphState, agents: dict) -> GraphState:
    """Узел для выполнения ВСЕХ исследовательских задач текущей фазы."""
    print("\n--- Узел: Research (Пакетная обработка фазы) ---")
    tasks = state['task_queue']
    researcher = agents['Researcher']
    contrarian = agents['Contrarian']
    
    raw_facts = []
    processed_tasks = []
    
    for task in tasks:
        # Пропускаем задачи, не являющиеся исследовательскими
        if task['agent_name'] not in ["Researcher", "Contrarian"]:
            continue
        
        try:
            model = state['model_assignments'][task['task_id']]
            if task['agent_name'] == 'Researcher':
                raw_facts.extend(researcher.execute(task, model))
            elif task['agent_name'] == 'Contrarian':
                raw_facts.extend(contrarian.execute(task, model))
            
            task['status'] = 'SUCCESS'
        except Exception as e:
            task['status'] = 'FAILURE'
            task['error'] = str(e)
        
        processed_tasks.append(task)

    state['node_outputs'] = {'raw_facts': raw_facts, 'processed_tasks': processed_tasks}
    return state

def qa_node(state: GraphState, assessor: QualityAssessorAgent) -> GraphState:
    """Узел для первичной и повторной оценки качества фактов."""
    print("\n--- Узел: Quality Assessment ---")
    facts_to_assess = state['node_outputs'].get('facts_for_assessment', [])
    if not facts_to_assess:
        print("   [QANode] Нет фактов для оценки.")
        state['node_outputs']['good'] = []
        state['node_outputs']['fixable'] = []
        return state

    model = "gemma-3" # Всегда используем дешевую модель для оценки
    assessment_report = assessor.execute(facts_to_assess, model)
    
    good, fixable = [], []
    assessments = {item['claim_id']: item for item in assessment_report.get('assessments', [])}
    
    for fact in facts_to_assess:
        assessment = assessments.get(fact['claim_id'])
        if assessment:
            if assessment['is_ok']:
                good.append(fact)
            elif assessment['is_fixable']:
                fact['feedback'] = assessment['reason']
                fixable.append(fact)
    
    state['node_outputs']['good'] = good
    state['node_outputs']['fixable'] = fixable
    return state

def fixer_node(state: GraphState, fixer: FixerAgent) -> GraphState:
    """Узел для исправления фактов из 'серой зоны'."""
    print("\n--- Узел: Fixer ---")
    facts_to_fix = state['node_outputs'].get('fixable', [])
    if not facts_to_fix:
        state['node_outputs']['fixed'] = []
        return state
        
    model = "gemma-3"
    fixed_facts = fixer.execute(facts_to_fix, model)
    state['node_outputs']['fixed'] = fixed_facts
    return state

def sanity_check_node(state: GraphState, critic: SanityCheckCritic) -> GraphState:
    """Узел для финальной проверки на здравый смысл."""
    print("\n--- Узел: Sanity Check ---")
    candidates = state['node_outputs'].get('candidates_for_sanity_check', [])
    if not candidates:
        state['node_outputs']['final_facts'] = []
        return state
    
    model = "gemini-2.5-flash"
    final_facts = critic.execute(candidates, model)
    state['node_outputs']['final_facts'] = final_facts
    return state

def commit_node(state: GraphState) -> GraphState:
    """Узел для сохранения верифицированных фактов в Базу Знаний."""
    print("\n--- Узел: Commit to KB ---")
    final_facts = state['node_outputs'].get('final_facts', [])
    current_kb = state.get('knowledge_base', {})
    for fact in final_facts:
        current_kb[fact['claim_id']] = fact
    state['knowledge_base'] = current_kb
    print(f"   [CommitNode] -> Добавлено/обновлено {len(final_facts)} фактов в Базе Знаний.")
    
    # Перемещаем все задачи из очереди в выполненные
    state['completed_tasks'].extend(state['task_queue'])
    state['task_queue'] = []
    return state

def janitor_node(state: GraphState, janitor: KnowledgeJanitorAgent) -> GraphState:
    """Узел для очистки и архивации Базы Знаний."""
    print("\n--- Узел: Janitor ---")
    state['knowledge_base'] = janitor.cleanup_knowledge_base(state['knowledge_base'])
    return state

def reflection_node(state: GraphState, analyst: AnalystAgent, supervisor: SupervisorAgent) -> GraphState:
    """Узел для анализа завершенной фазы и планирования следующей."""
    print("\n--- Узел: Reflection ---")
    analysis_result = analyst.execute_reflection(state['knowledge_base'], "gemini-2.5-flash")
    # ЗАГЛУШКА: Здесь должен быть вызов supervisor.create_next_phase_plan(analysis_result['data'])
    # который вернет список новых задач. Пока имитируем.
    print("   [Reflection] Генерация следующей фазы (логика-заглушка).")
    # state['task_queue'].extend(new_tasks)
    return state

def architect_node(state: GraphState, architect: ArchitectAgent) -> GraphState:
    """Узел для самокоррекции системы через мета-агента."""
    print("\n--- Узел: Architect ---")
    task = state['current_task']
    error = state.get('error_message', 'Нет деталей')
    fixed_task = architect.fix_task(task, error)
    state['task_queue'].insert(0, fixed_task)
    print(f"   [ArchitectNode] Задача {task['task_id']} исправлена и возвращена в очередь.")
    return state

def final_report_node(state: GraphState, analyst: AnalystAgent, writer: ReportWriterAgent, output_dir: str) -> GraphState:
    """Узел для генерации финального отчета."""
    print("\n--- Узел: Final Report ---")
    print("   [FinalReportNode] Запускаю финальный конвейер...")
    
    analysis_data = analyst.execute_final_synthesis(state['knowledge_base'], "gemini-2.5-flash")
    if not analysis_data:
        print("!!! [FinalReportNode] Ошибка на этапе анализа.")
        return state
        
    report_content = writer.execute(analysis_data, "gemma-3")
    if not report_content:
        print("!!! [FinalReportNode] Ошибка на этапе написания отчета.")
        return state

    final_markdown = citation_post_processor(report_content, state['knowledge_base'])
    
    report_path = os.path.join(output_dir, "Final_Report_v4.1.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)
    print(f"   [FinalReportNode] -> Финальный отчет сохранен в {report_path}")
    return state

# ====================================================================================
# === 2. ОПРЕДЕЛЕНИЕ МАРШРУТИЗАТОРОВ (CONDITIONAL EDGES) =============================
# ====================================================================================

def main_router(state: GraphState) -> str:
    """Главный маршрутизатор, управляющий потоком выполнения."""
    print("\n--- Узел: Main Router ---")
    
    # Логика эскалации (пока не реализована, т.к. research_node обрабатывает всю фазу)
    # Здесь можно добавить проверку статуса processed_tasks из research_node
    
    if state['task_queue']:
        print("   [MainRouter] -> Есть задачи. Запускаю исследовательский узел.")
        return "research"
    else:
        print("   [MainRouter] -> Задач нет. Фаза завершена. Перехожу к Janitor.")
        return "janitor"

def qa_router(state: GraphState) -> str:
    """Маршрутизатор для конвейера QA."""
    print("\n--- Узел: QA Router ---")
    if state['node_outputs'].get('fixable'):
        print("   [QARouter] -> Есть факты для исправления. Перехожу к Fixer.")
        return "fixer"
    else:
        print("   [QARouter] -> Нет фактов для исправления. Перехожу к финальной проверке.")
        # Собираем кандидатов для Sanity Check (только те, что были хороши изначально)
        state['node_outputs']['candidates_for_sanity_check'] = state['node_outputs'].get('good', [])
        return "sanity_check"

def reassessment_router(state: GraphState) -> str:
    """Маршрутизатор после повторной оценки исправленных фактов."""
    print("\n--- Узел: Re-assessment Router ---")
    # Собираем кандидатов: те, что были хороши изначально + те, что стали хорошими после исправления
    good_initial = state['node_outputs'].get('good_initial', [])
    fixed_reassessed_good = state['node_outputs'].get('good', []) # 'good' теперь содержит результат повторной оценки
    state['node_outputs']['candidates_for_sanity_check'] = good_initial + fixed_reassessed_good
    print(f"   [ReassessmentRouter] -> Всего кандидатов для финальной проверки: {len(state['node_outputs']['candidates_for_sanity_check'])}")
    return "sanity_check"

def reflection_router(state: GraphState) -> str:
    """Маршрутизатор после рефлексии."""
    print("\n--- Узел: Reflection Router ---")
    if state['task_queue']:
        print("   [ReflectionRouter] -> Обнаружены новые задачи. Начинаю следующую фазу.")
        return "research" # Начинаем новую фазу
    else:
        print("   [ReflectionRouter] -> Новых задач нет. План выполнен. Перехожу к финальному отчету.")
        return "final_report"

# ====================================================================================
# === 3. СБОРКА ГРАФА (WORKFLOW COMPILATION) =========================================
# ====================================================================================

def build_graph(agents: dict, output_dir: str):
    """Собирает и компилирует финальный граф LangGraph."""
    workflow = StateGraph(GraphState)

    # Добавление всех узлов
    workflow.add_node("supervisor", lambda state: supervisor_node(state, agents['Supervisor']))
    workflow.add_node("research", lambda state: research_node(state, agents))
    workflow.add_node("qa", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("fixer", lambda state: fixer_node(state, agents['Fixer']))
    workflow.add_node("qa_reassessment", lambda state: qa_node(state, agents['QualityAssessor']))
    workflow.add_node("sanity_check", lambda state: sanity_check_node(state, agents['SanityCheckCritic']))
    workflow.add_node("commit", commit_node)
    workflow.add_node("janitor", lambda state: janitor_node(state, agents['Janitor']))
    workflow.add_node("reflection", lambda state: reflection_node(state, agents['Analyst'], agents['Supervisor']))
    workflow.add_node("architect", lambda state: architect_node(state, agents['Architect']))
    workflow.add_node("final_report", lambda state: final_report_node(state, agents['Analyst'], agents['ReportWriter'], output_dir))

    # Определение логики графа
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", main_router, {"research": "research", "janitor": "janitor"})

    # Основной рабочий цикл
    workflow.add_edge("research", "qa")
    workflow.add_conditional_edges("qa", qa_router, {"fixer": "fixer", "sanity_check": "sanity_check"})
    workflow.add_edge("fixer", "qa_reassessment")
    workflow.add_conditional_edges("qa_reassessment", reassessment_router, {"sanity_check": "sanity_check"})
    workflow.add_edge("sanity_check", "commit")
    
    # Цикл завершения фазы и рефлексии
    workflow.add_edge("commit", "janitor")
    workflow.add_edge("janitor", "reflection")
    workflow.add_conditional_edges("reflection", reflection_router, {"research": "research", "final_report": "final_report"})

    # Ветка самокоррекции (пока не подключена к основному роутеру)
    # workflow.add_edge("architect", "research")

    workflow.add_edge("final_report", END)

    app = workflow.compile()
    print("-> Финальный граф вычислений v4.1 успешно скомпилирован.")
    return app