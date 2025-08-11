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
from core.context_compressor import ContextCompressor


# === ИЗМЕНЕНИЕ НАЧАТО: Импортируем ValidatorAgent ===
from agents.meta_agents import ArchitectAgent, KnowledgeJanitorAgent, ToolSmithAgent, ValidatorAgent, FailureAnalystAgent
# === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

from agents.supervisor import SupervisorAgent
from agents.workers import (
    OutlineAgent, ReportWriterAgent, SanityCheckCritic, SectionWriterAgent, SingleStepToolAgent, QualityAssessorAgent, FixerAgent, 
    AnalystAgent, ReviserAgent, FinancialModelAgent, ProductManagerAgent,
    ProductOwnerMemoAgent, InvestmentMemoAgent, CompetitorAnalysisAgent,
    TechnologyDeepDiveAgent, RoadmapVisualizationAgent
)
from tools.diagnostics import probe_google_api
# --- Константы ---
OUTPUT_DIR = "output"
STATE_FILE = os.path.join(OUTPUT_DIR, "graph_state.json")
CONFIG_FILE = "config.yaml"
API_LIMITS = {
    "gemini-2.5-pro": 100, "gemini-2.5-flash": 250, "gemini-2.5-flash-lite": 1000,
    "gemma-3": 14400, "gemma-3n": 14400, "gemini-embedding-001": 1000,
}
def run_pre_flight_checks():
    """
    Выполняет серию диагностических тестов перед запуском основного графа.
    """
    print("\n--- ЗАПУСК ПРЕДПОЛЕТНОЙ ПРОВЕРКИ API ---")
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("!!! КРИТИЧЕСКАЯ ОШИБКА: GOOGLE_API_KEY не найден в .env файле. Проверка невозможна.")
        return False

    # Модели, которые мы хотим проверить. Pro - критически важна. Flash - для сравнения.
    models_to_probe = ["gemini-2.5-pro", "gemini-2.5-flash"]
    all_checks_passed = True

    for model in models_to_probe:
        print("\n" + "="*30 + f" ПРОВЕРКА МОДЕЛИ: {model} " + "="*30)
        result = probe_google_api(model, api_key)
        
        print(f"  - Статус запроса: {result['status']}")
        print(f"  - HTTP Статус-код: {result['http_status_code']}")
        
        # Выводим тело ответа в удобном формате
        print("  - Тело ответа:")
        if isinstance(result['response_body'], dict):
            print(json.dumps(result['response_body'], indent=2, ensure_ascii=False))
        else:
            print(result['response_body'])
            
        if result['error_message']:
            print(f"  - Сообщение об ошибке: {result['error_message']}")

        # Критическая проверка для основной модели
        if model == "gemini-2.5-pro" and result['http_status_code'] != 200:
            all_checks_passed = False
            print("\n!!! КРИТИЧЕСКИЙ СБОЙ ПРОВЕРКИ! Основная модель 'gemini-2.5-pro' недоступна.")
            print("    Возможные причины:")
            print("    1. Неверный GOOGLE_API_KEY.")
            print("    2. Vertex AI API / Generative Language API не активирован в вашем проекте Google Cloud.")
            print("    3. К проекту Google Cloud не привязан платежный аккаунт.")
            print("    4. Исчерпаны квоты или лимиты для данной модели.")

    print("\n--- ПРЕДПОЛЕТНАЯ ПРОВЕРКА ЗАВЕРШЕНА ---\n")
    return all_checks_passed

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

    context_compressor = ContextCompressor(llm_client, budget_manager)

    # === ИЗМЕНЕНИЕ НАЧАТО: Вызов предполетной проверки закомментирован ===
    # if not run_pre_flight_checks():
    #     print("!!! Предполетная проверка провалена. Запуск основного графа отменен.")
    #     exit(1) # Завершаем выполнение с кодом ошибки
    # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    # --- 2.1. Инициализация Агентов ---
    toolsmith = ToolSmithAgent(llm_client, budget_manager)
    validator = ValidatorAgent(llm_client, budget_manager, tool_registry)
    architect = ArchitectAgent(llm_client, budget_manager, tool_registry, toolsmith)
    failure_analyst = FailureAnalystAgent(llm_client, budget_manager, tool_registry)

    agents = {
        # Мета-агенты
        "Supervisor": SupervisorAgent(llm_client, budget_manager),
        "Architect": architect,
        "Validator": validator,
        "Janitor": KnowledgeJanitorAgent(llm_client, budget_manager),
        "FailureAnalyst": failure_analyst,

        # Аналитические агенты
        "Reviser": ReviserAgent(llm_client, budget_manager, context_compressor),
        "Analyst": AnalystAgent(llm_client, budget_manager, context_compressor),
        
        # Агенты-исполнители
        "SingleStepToolAgent": SingleStepToolAgent(llm_client, budget_manager, tool_registry),
        
        # Агенты конвейера качества
        "QualityAssessor": QualityAssessorAgent(llm_client, budget_manager),
        "Fixer": FixerAgent(llm_client, budget_manager),
        "SanityCheckCritic": SanityCheckCritic(llm_client, budget_manager),


        "OutlineAgent": OutlineAgent(llm_client, budget_manager),
        "SectionWriterAgent": SectionWriterAgent(llm_client, budget_manager),
        "ReportWriter": ReportWriterAgent(llm_client, budget_manager),

        "FinancialModelAgent": FinancialModelAgent(llm_client, budget_manager),
        "ProductManagerAgent": ProductManagerAgent(llm_client, budget_manager),
        
        # Новые агенты-специалисты
        "ProductOwnerMemoAgent": ProductOwnerMemoAgent(llm_client, budget_manager, context_compressor),
        "InvestmentMemoAgent": InvestmentMemoAgent(llm_client, budget_manager, context_compressor),
        "CompetitorAnalysisAgent": CompetitorAnalysisAgent(llm_client, budget_manager, context_compressor),
        "TechnologyDeepDiveAgent": TechnologyDeepDiveAgent(llm_client, budget_manager, context_compressor),
        "RoadmapVisualizationAgent": RoadmapVisualizationAgent(llm_client, budget_manager),
    }
    print(f"-> Пул агентов инициализирован. Всего зарегистрировано: {len(agents)} агентов.")

   # --- 3. Сборка графа и определение начального состояния ---
    app = build_graph(agents, OUTPUT_DIR)
    initial_state = GraphState(
        user_config=user_config,
        task_queue=[], completed_tasks=[], knowledge_base={}, artifacts={},
        model_assignments={}, visited_urls=[], escalation_count=0,
        current_task=None, error_message=None, node_outputs={},
        report_outline={}, drafted_sections=[], current_section_to_draft=None
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
            print(f"   [Main] РЕЖИМ: Новый план с сохранением Базы Знаний...")
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                saved_state = json.load(f)
            initial_state['knowledge_base'] = saved_state.get('knowledge_base', {})
            print("   [Main] <- База Знаний загружена. Генерирую новый план.")
        else:
            print(f"   [Main] РЕЖИМ: Новый запуск с контекстом из '{CONFIG_FILE}'.")
        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

    # --- 5. Запуск графа ---
    print("\n--- ЗАПУСК ГРАФА ВЫЧИСЛЕНИЙ v4.4 ---")
    run(app, initial_state, STATE_FILE)

if __name__ == "__main__":
    main()