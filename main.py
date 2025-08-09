# main.py
import os
from dotenv import load_dotenv
from core.state import GraphState
from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager
from agents.supervisor import SupervisorAgent
from agents.workers import ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, AnalystAgent, ReportWriterAgent
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ToolSmithAgent
import orchestrator

def main():
    load_dotenv()
    print("Инициализация системы 'Динамический Фреймворк v4.1'...")

    # 1. Инициализация базовых сервисов
    daily_limits = {
        "gemini-2.5-pro": 100, "gemini-2.5-flash": 250,
        "gemini-2.5-flash-lite": 1000, "gemma-3": 14400, "gemma-3n": 14400
    }
    budget_manager = APIBudgetManager("output", daily_limits)
    llm_client = LLMClient(budget_manager)

    # 2. Инициализация ВСЕХ агентов
    agents = {
        "Supervisor": SupervisorAgent(llm_client, budget_manager),
        "Researcher": ResearcherAgent(llm_client, budget_manager),
        "Contrarian": ContrarianAgent(llm_client, budget_manager),
        "QualityAssessor": QualityAssessorAgent(llm_client, budget_manager),
        "Fixer": FixerAgent(llm_client, budget_manager),
        "Analyst": AnalystAgent(llm_client, budget_manager),
        "ReportWriter": ReportWriterAgent(llm_client, budget_manager),
        "Architect": ArchitectAgent(llm_client, budget_manager),
        "Janitor": KnowledgeJanitorAgent(llm_client, budget_manager),
        "ToolSmith": ToolSmithAgent(llm_client, budget_manager),
    }
    
    # 3. Сборка графа
    app = orchestrator.build_graph(agents)

    # 4. Определение начального состояния
    initial_state: GraphState = {
        "task_queue": [], "completed_tasks": [], "knowledge_base": {},
        "current_task": None, "model_assignments": {}, "escalation_count": 0,
        "current_task_result": None, "error_message": None
    }

    # 5. Запуск
    orchestrator.run(app, initial_state)

if __name__ == "__main__":
    main()