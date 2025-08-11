# main.py
import os
import json
import argparse
from dotenv import load_dotenv
import yaml

from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager
from core.tool_registry import ToolRegistry
from core.state import GraphState
from orchestrator import build_graph, run

# === ИЗМЕНЕНИЕ НАЧАТО: Импортируем ValidatorAgent ===
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ToolSmithAgent, ValidatorAgent
# === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

from agents.supervisor import SupervisorAgent
from agents.workers import (
    ResearcherAgent, ContrarianAgent, QualityAssessorAgent, FixerAgent, 
    AnalystAgent, ReportWriterAgent, SanityCheckCritic,
    FinancialModelAgent, ProductManagerAgent, ReviserAgent,
    OutlineAgent, SectionWriterAgent
)

# --- Константы ---
OUTPUT_DIR = "output"
STATE_FILE = os.path.join(OUTPUT_DIR, "graph_state.json")
CONFIG_FILE = "config.yaml"
API_LIMITS = {
    "gemini-2.5-pro": 100, "gemini-2.5-flash": 250, "gemini-2.5-flash-lite": 1000,
    "gemma-3": 14400, "gemma-3n": 14400, "gemini-embedding-001": 1000,
}

def main():
    # --- 1. Загрузка конфигурации и настройка окружения ---
    load_dotenv()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="Запуск AI Factory v4.3")
    parser.add_argument('--new-plan-keep-kb', action='store_true', help="Пересоздать план, но сохранить Базу Знаний.")
    args = parser.parse_args()

    # --- 2. Инициализация всех компонентов ---
    print("Инициализация системы 'Динамический Фреймворк v4.3'...")
    
    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        user_config = yaml.safe_load(f)
    print(f"   [Main] Пользовательский конфиг '{CONFIG_FILE}' успешно загружен.")

    budget_manager = APIBudgetManager(OUTPUT_DIR, API_LIMITS)
    llm_client = LLMClient(budget_manager)
    tool_registry = ToolRegistry(generated_tools_dir="tools/generated")

    # --- 2.1. Инициализация Агентов ---
    toolsmith = ToolSmithAgent(llm_client, budget_manager)
    
    # === ИЗМЕНЕНИЕ НАЧАТО: Создаем экземпляр ValidatorAgent ===
    validator = ValidatorAgent(llm_client, budget_manager, tool_registry)
    # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    architect = ArchitectAgent(llm_client, budget_manager, tool_registry, toolsmith)
    
    agents = {
        "Supervisor": SupervisorAgent(llm_client, budget_manager),
        "Reviser": ReviserAgent(llm_client, budget_manager),
        "Researcher": ResearcherAgent(llm_client, budget_manager, tool_registry),
        "Contrarian": ContrarianAgent(llm_client, budget_manager, tool_registry),
        "QualityAssessor": QualityAssessorAgent(llm_client, budget_manager),
        "Fixer": FixerAgent(llm_client, budget_manager),
        "SanityCheckCritic": SanityCheckCritic(llm_client, budget_manager),
        "Analyst": AnalystAgent(llm_client, budget_manager),
        "OutlineAgent": OutlineAgent(llm_client, budget_manager),
        "SectionWriterAgent": SectionWriterAgent(llm_client, budget_manager),
        "ReportWriter": ReportWriterAgent(llm_client, budget_manager),
        "Janitor": KnowledgeJanitorAgent(llm_client, budget_manager),
        "FinancialModelAgent": FinancialModelAgent(llm_client, budget_manager),
        "ProductManagerAgent": ProductManagerAgent(llm_client, budget_manager),
        "Architect": architect,
        # === ИЗМЕНЕНИЕ НАЧАТО: Добавляем валидатора в словарь ===
        "Validator": validator
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===
    }

    # --- 3. Сборка графа и определение начального состояния ---
    app = build_graph(agents, OUTPUT_DIR)
    initial_state = GraphState(
        user_config=user_config,
        task_queue=[],
        completed_tasks=[],
        knowledge_base={},
        artifacts={},
        model_assignments={},
        visited_urls=[],
        escalation_count=0,
        current_task=None,
        error_message=None,
        node_outputs={},
        report_outline={},
        drafted_sections=[],
        current_section_to_draft=None
    )

    # --- 4. Логика возобновления / нового запуска ---
    if os.path.exists(STATE_FILE) and not args.new_plan_keep_kb:
        print(f"   [Main] РЕЖИМ: Продолжение. Загружаю состояние из '{STATE_FILE}'...")
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            saved_state = json.load(f)
        initial_state.update(saved_state)
        print("   [Main] <- Состояние успешно загружено. Возобновляю работу.")
    else:
        if args.new_plan_keep_kb and os.path.exists(STATE_FILE):
            print(f"   [Main] РЕЖИМ: Новый план с сохранением Базы Знаний. Загружаю KB из '{STATE_FILE}'...")
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
            initial_state['knowledge_base'] = saved_state.get('knowledge_base', {})
            print("   [Main] <- База Знаний загружена. Генерирую новый план.")
        else:
            print(f"   [Main] РЕЖИМ: Новый запуск. Создаю новую сессию с контекстом из '{CONFIG_FILE}'.")
        # Удаляем старый файл состояния, если он есть, чтобы начать с чистого листа
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    # --- 5. Запуск графа ---
    print("\n--- ЗАПУСК ГРАФА ВЫЧИСЛЕНИЙ v4.3 ---")
    run(app, initial_state, STATE_FILE)

if __name__ == "__main__":
    main()