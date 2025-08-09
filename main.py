# main.py
import os
import time
import argparse
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.api_core.exceptions import ResourceExhausted

from core.world_model import WorldModel
from core.budget_manager import APIBudgetManager
from agents.search_agent import SearchAgent
from utils.helpers import SearchAPIFailureError, citation_post_processor # ИМПОРТ НОВОЙ ФУНКЦИИ

from agents.supervisor import SupervisorAgent
from agents.researcher import ResearcherAgent
from agents.contrarian import ContrarianAgent
from agents.quality_assessor import BatchQualityAssessor
from agents.fixer import BatchFixerAgent
from agents.analyst import AnalystAgent # ИМПОРТ АНАЛИТИКА
from agents.report_writer import ReportWriterAgent

TASK_BATCH_SIZE = 2
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 60
TASK_COOLDOWN_SECONDS = 5
STRATEGIST_COOLDOWN_SECONDS = 10

def main():
    parser = argparse.ArgumentParser(description="Фабрика Обогащения Данных")
    parser.add_argument('--fresh-start', action='store_true')
    parser.add_argument('--new-plan-keep-kb', action='to_plan_only')
    args = parser.parse_args()

    load_dotenv()
    print("Инициализация системы 'Фабрика Обогащения Данных' v3.0...")

    try:
        llms = {
            "pro": ChatGoogleGenerativeAI(model="models/gemini-2.5-pro", temperature=0.3),
            "flash": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0.1),
            "lite": ChatGoogleGenerativeAI(model="models/gemini-2.5-flash-lite", temperature=0.1),
            "gemma": ChatGoogleGenerativeAI(model="models/gemma-3-27b-it", temperature=0.0),
        }
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать модели LLM: {e}")
        return

    # Инициализация WorldModel и BudgetManager
    daily_limits = {
        "models/gemini-2.5-pro": 100, "models/gemini-2.5-flash": 250,
        "models/gemini-2.5-flash-lite": 1000, "models/gemma-3-27b-it": 14400,
        "models/gemini-embedding-001": 1000,
    }
    output_directory = "output"
    budget_manager = APIBudgetManager(output_directory, daily_limits)
    world_model = WorldModel(
        static_context={
            # ... (ваш static_context без изменений) ...
             "project_owner_profile": """
- **Роль и Цель:**
  - **Целевая Роль:** Руководитель AI-продуктов / AI Product Owner.
  - **Цель Проекта:** Создать убедительную концепцию нового HR-Tech продукта, чтобы доказать свою компетенцию и возглавить это новое бизнес-направление.
- **Технический Стек и Навыки:**
  - **Языки и Фреймворки:** Python (уверенно), FastAPI/Flask, SQL/NoSQL.
  - **AI/ML:** LangChain, TensorFlow, PyTorch, Scikit-learn, Pandas. Опыт fine-tuning и применения LLM.
- **Ключевые Компетенции:**
  - **Full-Cycle ML Development:** Способность вести проект от сбора данных до создания работающего веб-прототипа.
  - **Продуктовое Мышление:** Опыт быстрой проверки гипотез и создания MVP.
  - **R&D и Системный Анализ:** Опыт работы в рамках научных проектов, требующих глубокого анализа.
- **Зоны Роста:**
  - **Инфраструктура:** Не являюсь экспертом в DevOps. Фокус — на логике и моделях.
  - **Frontend:** Не являюсь Frontend-разработчиком.
  - **Стиль Работы:** Наиболее эффективен в роли, требующей интеллектуального лидерства (анализ, стратегия, архитектура ИИ).
""",
            "initial_brief": """
**Ключевое Видение Продукта (основано на предварительной записке, получившей положительный отклик):**
1.  **Трансформация Обучения в Карьерный Рост (Идея: "Карьерный Навигатор"):** Вместо статичного списка курсов, предложить сотруднику интерактивную "Карту Карьерной Траектории".
2.  **Персонализация и Поддержка в Масштабе (Идея: "AI-Агент/Ментор"):** Создать интеллектуального помощника (возможно на базе бота Expecto Patronum), который анализирует данные из всех модулей и выступает в роли персонального гида.
3.  **Новая Бизнес-Ценность (Идея: "Talent-as-a-Service" / "Внутренний Рынок Талантов"):** Превратить платформу в "кадровый инкубатор", где руководители видят "созревание" специалистов.
""",
            "tsu_assets_analysis": """
**ВХОДНОЙ БРИФ ДЛЯ ВЕРИФИКАЦИИ И УГЛУБЛЕНИЯ:**
**1. Сводный Анализ Продуктового Портфеля ТГУ**
*   **Ключевая Синергия:** Портфель представляет собой взаимодополняющую экосистему с технологическим ядром в виде платформы LMS IDO (модифицированный Moodle). Остальные продукты (ИИ-оценщик, Цифровой Репетитор) спроектированы как интегрируемые модули.
*   **Текущие Целевые Рынки:** Продукты преимущественно ориентированы на сектор образования (B2Edu) и госсектор (B2G). Выход на корпоративный B2B HR-рынок является гипотезой.
*   **Технологическое Ядро:** Moodle (LMS IDO) и собственные наработки в области прикладного ИИ.
**2. Детальный Разбор Ключевых Активов для B2B HR-Tech**
*   **Продукт: ИИ-оценщик**
    *   **Сильные Стороны:** Автоматизирует проверку работ, дает детальную обратную связь.
    *   **Слабые Стороны:** Ограничен гуманитарными текстами, требует "постпроверки" преподавателем, является плагином для Moodle.
    *   **Оценка Готовности к B2B HR: 3/10.**
*   **Продукт: LMS IDO**
    *   **Сильные Стороны:** Готовая LMS на базе Moodle с техподдержкой, зарегистрирована в реестре ПО РФ.
    *   **Слабые Стороны:** Устаревший UI/UX, риски производительности, академическая направленность.
    *   **Оценка Готовности к B2B HR: 5/10.**
**3. Анализ Второстепенных и Скрытых Активов**
*   **Telegram-бот 'Expecto Patronum':** Доказывает наличие компетенций в создании чат-ботов.
*   **Проект "РосНавык":** Демонстрирует способность анализировать рынок труда.
*   **Проект "Кампусная карта" на блокчейне:** Демонстрирует R&D-компетенции в области верифицируемых креденшелов.
**4. Вывод:** Наиболее жизнеспособной стратегией является не продажа существующих продуктов "как есть", а их глубокая пересборка и комбинация для создания нового, рыночно-ориентированного B2B HR-продукта.
**ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ДЛЯ СИСТЕМЫ:**
- Необходимо провести открытый поиск других EdTech-проектов, стартапов и цифровых инициатив, связанных с ТГУ.
""",
            "project_location": {
                "city": "Томск, Россия",
                "comment": "С высокой вероятностью (90%) основная команда разработки будет находиться здесь. Финансовые расчеты (зарплаты, аренда) должны в первую очередь учитывать этот регион. Если данных по региону нет, использовать данные по Москве/РФ с предложением поправочного коэффициента."
            },
            "main_goal": "Подготовить высококачественную, убедительную аналитическую записку для коммерческого директора ТГУ, предлагающую концепцию нового продукта в рамках существующей экосистемы iDO. **Цель записки — не просто предоставить информацию, а продемонстрировать мою способность взять на себя роль Владельца Продукта (Product Owner) для этого конкретного проекта, отвечая за его концепцию, логику и развитие от идеи до прототипа.**"
        },
        budget_manager=budget_manager,
        output_dir=output_directory,
        force_fresh_start=args.fresh_start,
        reset_plan_only=args.new_plan_keep_kb
    )

    # Инициализация агентов
    search_agent = SearchAgent(serper_api_key=os.getenv("SERPER_API_KEY"), cache_dir=world_model.cache_dir)
    supervisor = SupervisorAgent(llm=llms["pro"], sanitizer_llm=llms["gemma"], budget_manager=budget_manager)
    researcher = ResearcherAgent(llm=llms["flash"], sanitizer_llm=llms["lite"], search_agent=search_agent, budget_manager=budget_manager)
    contrarian = ContrarianAgent(llm=llms["flash"], sanitizer_llm=llms["lite"], search_agent=search_agent, budget_manager=budget_manager)
    quality_assessor = BatchQualityAssessor(llm=llms["lite"], sanitizer_llm=llms["lite"], budget_manager=budget_manager)
    fixer = BatchFixerAgent(llm=llms["flash"], sanitizer_llm=llms["lite"], budget_manager=budget_manager)
    analyst = AnalystAgent(llm=llms["pro"], sanitizer_llm=llms["gemma"], budget_manager=budget_manager) # ИНИЦИАЛИЗАЦИЯ АНАЛИТИКА
    report_writer = ReportWriterAgent(llm=llms["pro"], sanitizer_llm=llms["gemma"], budget_manager=budget_manager)

    # Создание плана
    if not world_model.get_full_context()['dynamic_knowledge']['strategic_plan']:
        plan = supervisor.create_strategic_plan(world_model.get_full_context())
        if plan and plan.get("phases"):
            plan["phases"][0]["status"] = "IN_PROGRESS"
        world_model.update_strategic_plan(plan)
        world_model.save_state()

    # Главный цикл
    while True:
        active_tasks = world_model.get_active_tasks()

        if not active_tasks:
            print("\n--- Все задачи выполнены. Запускаю декомпозированную рефлексию... ---")
            
            # ЭТАП 1: АНАЛИЗ
            kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
            analyst_report = analyst.run_reflection_analysis(kb)
            
            # ЭТАП 2: ПЛАНИРОВАНИЕ
            current_plan = world_model.get_full_context()['dynamic_knowledge']['strategic_plan']
            updated_plan = supervisor.reflect_and_update_plan(analyst_report, current_plan)
            world_model.update_strategic_plan(updated_plan)

            if world_model.get_full_context()['dynamic_knowledge']['strategic_plan'].get("main_goal_status") == "READY_FOR_FINAL_BRIEF":
                break
            if not world_model.get_active_tasks():
                 print("!!! [Оркестратор] КРИТИЧЕСКАЯ ОШИБКА: Рефлексия не привела к созданию новых задач.")
                 break
            world_model.save_state()
            time.sleep(STRATEGIST_COOLDOWN_SECONDS)
            continue

        # Пакетная обработка задач (без изменений)
        tasks_to_run_batch = active_tasks[:TASK_BATCH_SIZE * 2]
        tasks_by_assignee = defaultdict(list)
        for task in tasks_to_run_batch:
            tasks_by_assignee[task['assignee']].append(task)
        
        all_raw_claims, processed_task_ids = [], set()
        try:
            researcher_tasks = tasks_by_assignee.get('ResearcherAgent', [])
            if researcher_tasks:
                all_raw_claims.extend(researcher.execute_batch(researcher_tasks))
                processed_task_ids.update([t['task_id'] for t in researcher_tasks])

            contrarian_tasks = tasks_by_assignee.get('ContrarianAgent', [])
            if contrarian_tasks:
                all_raw_claims.extend(contrarian.execute_batch(contrarian_tasks))
                processed_task_ids.update([t['task_id'] for t in contrarian_tasks])
            
            if not processed_task_ids:
                time.sleep(TASK_COOLDOWN_SECONDS)
                continue

            assessment = quality_assessor.assess_batch(all_raw_claims)
            fixed_claims = fixer.fix_batch(assessment['fixable_claims'])
            verified_claims = assessment['good_claims'] + fixed_claims
            
            if verified_claims:
                world_model.add_claims_to_kb(verified_claims)

            for task_id in processed_task_ids:
                world_model.update_task_status(task_id, 'COMPLETED')
                world_model.log_transaction({'task': {'task_id': task_id}, 'results': "Processed in batch."})

        except (SearchAPIFailureError, ResourceExhausted) as e:
            print(f"!!! ОРКЕСТРАТОР: Системная ошибка при выполнении пакета: {e}")
            for task_id in processed_task_ids: world_model.update_task_status(task_id, 'FAILED')
        except Exception as e:
            print(f"!!! ОРКЕСТРАТОР: Непредвиденная ошибка при выполнении пакета: {e}")
            for task_id in processed_task_ids: world_model.update_task_status(task_id, 'FAILED')

        print("   [Оркестратор] Фиксирую изменения на диске...")
        world_model.save_state()
        time.sleep(TASK_COOLDOWN_SECONDS)

    # --- НОВЫЙ КОНВЕЙЕР ГЕНЕРАЦИИ ФИНАЛЬНОГО ОТЧЕТА ---
    print("\n--- Запускаю конвейер генерации финального отчета ---")
    
    # 1. Синтез
    kb = world_model.get_full_context()['dynamic_knowledge']['knowledge_base']
    synthesis_data = analyst.run_final_synthesis(kb)
    
    # 2. Написание
    raw_markdown = report_writer.write_final_report(synthesis_data)
    
    # 3. Детерминированная пост-обработка
    final_markdown = citation_post_processor(raw_markdown, kb)
    
    # 4. Сохранение
    report_path = os.path.join(world_model.output_dir, "Executive_Summary_For_Director.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)
    print(f"-> Финальный отчет с верифицированными цитатами сохранен в {report_path}")

    print(f"\n--- РАБОТА УСПЕШНО ЗАВЕРШЕНА ---")

if __name__ == "__main__":
    main()