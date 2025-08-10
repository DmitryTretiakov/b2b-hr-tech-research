# main.py
import os
import json
import yaml
import argparse
from dotenv import load_dotenv
from core.state import GraphState
from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager
from core.tool_registry import ToolRegistry
from agents.supervisor import SupervisorAgent
from agents.workers import (
    ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, 
    AnalystAgent, ReportWriterAgent, SanityCheckCritic,
    FinancialModelAgent, ProductManagerAgent
)
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ToolSmithAgent
import orchestrator

def main():
    # 1. Настройка аргументов командной строки
    parser = argparse.ArgumentParser(description="Запуск системы 'Динамический Фреймворк v4.2'.")
    parser.add_argument(
        '--new-plan-keep-kb',
        action='store_true',
        help='Начать выполнение с новым планом, но сохранить Базу Знаний из предыдущей сессии.'
    )
    args = parser.parse_args()

    load_dotenv()
    print("Инициализация системы 'Динамический Фреймворк v4.2'...")

    # 2. Загрузка пользовательского контекста
    try:
        with open("config.yaml", "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f)
        print("   [Main] Пользовательский конфиг 'config.yaml' успешно загружен.")
    except (IOError, yaml.YAMLError) as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить 'config.yaml'. {e}")
        return

    # 3. Инициализация базовых сервисов
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    daily_limits = { "gemini-2.5-pro": 100, "gemini-2.5-flash": 250, "gemini-2.5-flash-lite": 1000, "gemma-3": 14400, "gemma-3n": 14400, "gemini-embedding-001": 1000 }
    budget_manager = APIBudgetManager(output_dir, daily_limits)
    llm_client = LLMClient(budget_manager)
    tool_registry = ToolRegistry(generated_tools_dir=os.path.join(output_dir, "generated_tools"))
    
    # 4. Инициализация ВСЕХ агентов с внедрением зависимостей
    tool_smith = ToolSmithAgent(llm_client, budget_manager)
    architect = ArchitectAgent(llm_client, budget_manager, tool_smith, tool_registry)
    
    agents = {
        "Supervisor": SupervisorAgent(llm_client, budget_manager, tool_registry),
        "Researcher": ResearcherAgent(llm_client, budget_manager, tool_registry),
        "Contrarian": ContrarianAgent(llm_client, budget_manager, tool_registry),
        "QualityAssessor": QualityAssessorAgent(llm_client, budget_manager),
        "Fixer": FixerAgent(llm_client, budget_manager),
        "SanityCheckCritic": SanityCheckCritic(llm_client, budget_manager),
        "Analyst": AnalystAgent(llm_client, budget_manager, tool_registry),
        "ReportWriter": ReportWriterAgent(llm_client, budget_manager),
        "Architect": architect,
        "Janitor": KnowledgeJanitorAgent(llm_client, budget_manager),
        "ToolSmith": tool_smith,
        "FinancialModelAgent": FinancialModelAgent(llm_client, budget_manager),
        "ProductManagerAgent": ProductManagerAgent(llm_client, budget_manager),
    }
    
    # 5. Сборка графа
    app = orchestrator.build_graph(agents, output_dir)
    
    # 6. Определение состояния для запуска
    state_file_path = os.path.join(output_dir, "graph_state.json")
    
    initial_state_template: GraphState = {
        "user_config": user_config, "task_queue": [], "completed_tasks": [], "knowledge_base": {},
        "artifacts": {}, "current_task": None, "model_assignments": {}, "escalation_count": 0,
        "error_message": None, "node_outputs": {}
    }

    if args.new_plan_keep_kb and os.path.exists(state_file_path):
        print(f"   [Main] РЕЖИМ: Новый план с сохранением Базы Знаний.")
        with open(state_file_path, "r", encoding="utf-8") as f:
            old_state = json.load(f)
        state_to_run = initial_state_template
        state_to_run["knowledge_base"] = old_state.get("knowledge_base", {})
        print(f"   [Main] <- База Знаний ({len(state_to_run['knowledge_base'])} фактов) перенесена в новую сессию.")
    elif os.path.exists(state_file_path):
        print(f"   [Main] РЕЖИМ: Продолжение. Загружаю состояние из '{state_file_path}'...")
        with open(state_file_path, "r", encoding="utf-8") as f:
            state_to_run = json.load(f)
        print("   [Main] <- Состояние успешно загружено. Возобновляю работу.")
    else:
        print("   [Main] РЕЖИМ: Новый запуск. Создаю новую сессию с контекстом из 'config.yaml'.")
        state_to_run = initial_state_template

    # 7. Запуск
    print("\n--- ЗАПУСК ГРАФА ВЫЧИСЛЕНИЙ v4.2 ---")
    orchestrator.run(app, state_to_run, state_file_path)

if __name__ == "__main__":
    main()