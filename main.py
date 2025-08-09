# main.py
import os
from dotenv import load_dotenv
from core.state import GraphState
from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager
from agents.supervisor import SupervisorAgent
from agents.workers import ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, AnalystAgent, ReportWriterAgent, SanityCheckCritic
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ToolSmithAgent
import orchestrator

def main():
    load_dotenv()
    print("Инициализация системы 'Динамический Фреймворк v4.2'...")

    # 1. Инициализация базовых сервисов
    daily_limits = {
        "gemini-2.5-pro": 100,
        "gemini-2.5-flash": 250,
        "gemini-2.5-flash-lite": 1000,
        "gemma-3": 14400,
        "gemma-3n": 14400,
        "gemini-embedding-001": 1000,
    }
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    budget_manager = APIBudgetManager(output_dir, daily_limits)
    llm_client = LLMClient(budget_manager)

    # 2. Инициализация ВСЕХ агентов
    agents = {
        "Supervisor": SupervisorAgent(llm_client, budget_manager),
        "Researcher": ResearcherAgent(llm_client, budget_manager),
        "Contrarian": ContrarianAgent(llm_client, budget_manager),
        "QualityAssessor": QualityAssessorAgent(llm_client, budget_manager),
        "Fixer": FixerAgent(llm_client, budget_manager),
        "SanityCheckCritic": SanityCheckCritic(llm_client, budget_manager),
        "Analyst": AnalystAgent(llm_client, budget_manager),
        "ReportWriter": ReportWriterAgent(llm_client, budget_manager),
        "Architect": ArchitectAgent(llm_client, budget_manager),
        "Janitor": KnowledgeJanitorAgent(llm_client, budget_manager),
        "ToolSmith": ToolSmithAgent(llm_client, budget_manager),
    }
    
    # 3. Сборка графа
    app = orchestrator.build_graph(agents, output_dir)

    # 4. Определение начального состояния
    # Этот объект должен точно соответствовать структуре GraphState
    initial_state: GraphState = {
        "task_queue": [],
        "completed_tasks": [],
        "knowledge_base": {},
        "current_task": None,
        "model_assignments": {},
        "escalation_count": 0,
        "error_message": None,
        "node_outputs": {}
    }

    # 5. Запуск
    print("\n--- ЗАПУСК ГРАФА ВЫЧИСЛЕНИЙ v4.2 ---")
    # Примечание: world_model (сохранение/загрузка состояния) будет интегрирован позже.
    # Сейчас мы запускаем граф с чистого листа.
    orchestrator.run(app, initial_state)

if __name__ == "__main__":
    main()