# main.py
import json
import os
import sys
import time 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import argparse
from core.world_model import WorldModel
from core.budget_manager import APIBudgetManager
from agents.chief_strategist import ChiefStrategist
from agents.expert_team import ExpertTeam
from agents.search_agent import SearchAgent
from utils.helpers import SearchAPIFailureError
from google.api_core.exceptions import ResourceExhausted

# --- КОНСТАНТЫ ДЛЯ УПРАВЛЕНИЯ СКОРОСТЬЮ И ПОПЫТКАМИ ---
MAX_RETRIES = 3
# Увеличиваем задержку при ошибке API до 5 минут, чтобы дать сервису "остыть"
RETRY_DELAY_SECONDS = 300 

# --- НОВЫЕ КОНСТАНТЫ ДЛЯ ДРОССЕЛИРОВАНИЯ ---
# Задержка после КАЖДОЙ выполненной задачи для снижения общей частоты запросов
TASK_COOLDOWN_SECONDS = 15 
# Задержка после ресурсоемкой операции рефлексии у Стратега
STRATEGIST_COOLDOWN_SECONDS = 30 

def main():
    parser = argparse.ArgumentParser(description="Автономный Проектный Офис")
    parser.add_argument(
        '--fresh-start',
        action='store_true',
        help='Начать новое исследование, игнорируя сохраненное состояние.'
    )

    
    
    
    args = parser.parse_args()
    # --- ИНИЦИАЛИЗАЦИЯ ---
    load_dotenv()
    print("Инициализация системы 'Автономный Проектный Офис'...")
    
    # --- ДИАГНОСТИКА: ПРОВЕРКА ЗАГРУЗКИ КЛЮЧЕЙ ---
    print("\n--- ДИАГНОСТИКА API КЛЮЧЕЙ ---")
    serper_key = os.getenv("SERPER_API_KEY")
    google_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    
    if serper_key:
        print(f"   [OK] SERPER_API_KEY загружен. Длина: {len(serper_key)}, Первые 4 символа: {serper_key[:4]}")
    else:
        print("   [!!! ОШИБКА] SERPER_API_KEY НЕ найден!")
        
    if google_key:
        print(f"   [OK] GOOGLE_SEARCH_API_KEY загружен.")
    else:
        print("   [!!! ВНИМАНИЕ] GOOGLE_SEARCH_API_KEY НЕ найден.")
    print("---------------------------------\n")
    # --- КОНЕЦ ДИАГНОСТИКИ ---

    # ИНИЦИАЛИЗАЦИЯ LLM С ТОНКОЙ НАСТРОЙКОЙ ТЕМПЕРАТУРЫ
    try:
        llms = {
            "strategist": ChatGoogleGenerativeAI(
                model="models/gemini-2.5-pro",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.3 # Низкая температура для стратегической точности
            ),
            "expert_flash": ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1 # Очень низкая температура для надежной генерации JSON и аудита
            ),
            "expert_lite": ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash-lite",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.1 # Очень низкая температура для надежной генерации JSON
            ),
            # --- НОВАЯ МОДЕЛЬ ДЛЯ АУДИТА ИСТОЧНИКОВ ---
            "source_auditor": ChatGoogleGenerativeAI(
                model="models/gemma-3-27b-it",
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.0 # Нулевая температура для максимальной предсказуемости в задачах классификации
            )
        }
        print("-> Модели LLM успешно инициализированы с оптимальными настройками температуры.")
    except Exception as e:
        print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось инициализировать модели LLM. Проверьте GOOGLE_API_KEY. Ошибка: {e}")
        return
    
    daily_limits = {
        "models/gemini-2.5-pro": 100,
        "models/gemini-2.5-flash": 250,
        "models/gemini-2.5-flash-lite": 1000,
        "models/gemini-2.0-flash": 200,
        "models/gemini-2.0-flash-lite": 200,
        "models/gemma-3-27b-it": 14400,
        "models/gemini-embedding-001": 1000,
        "models/gemma-3-12b-it": 14400
    }
    # Определяем путь к output заранее, чтобы передать его обоим модулям
    output_directory = "output"
    budget_manager = APIBudgetManager(output_directory, daily_limits)

    world_model = WorldModel(
        static_context={
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
        force_fresh_start=args.fresh_start 
    )

    search_agent = SearchAgent(
        google_api_key=os.getenv("GOOGLE_SEARCH_API_KEY"),
        google_cx_id=os.getenv("SEARCH_ENGINE_ID"),
        serper_api_key=os.getenv("SERPER_API_KEY"),
        cache_dir=world_model.cache_dir
    )


    expert_team = ExpertTeam(llms, search_agent, budget_manager)
    strategist = ChiefStrategist(llm=llms["strategist"], medium_llm=llms["expert_flash"], sanitizer_llm=llms["source_auditor"], budget_manager=budget_manager)
    
    # --- ЗАПУСК РАБОТЫ ---
    print("\n--- ЗАПУСК ОСНОВНОГО ЦИКЛА ОРКЕСТРАТОРА ---")

    # --- СОЗДАЕМ ПЛАН, ТОЛЬКО ЕСЛИ ЕГО НЕТ ---
    # Если мы не начинаем с чистого листа и план уже есть, мы его не перезаписываем.
    if not world_model.get_full_context()['dynamic_knowledge']['strategic_plan']:
        print("\n--- Стратегический план не найден. Создаю новый... ---")
        plan = strategist.create_strategic_plan(world_model.get_full_context())
        
        # --- НАЧАЛО ИСПРАВЛЕНИЯ: АКТИВАЦИЯ ПЕРВОЙ ФАЗЫ ---
        if plan and plan.get("phases"):
            print("--- Активирую первую фазу плана... ---")
            plan["phases"][0]["status"] = "IN_PROGRESS"
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        world_model.update_strategic_plan(plan)
    else:
        print("\n--- Обнаружен существующий стратегический план. Продолжаю работу... ---")

    while True:
        active_tasks = world_model.get_active_tasks()
        
        if not active_tasks:
            print("\n--- Все задачи текущей фазы выполнены. Запускаю рефлексию Стратега... ---")
            
            reflection_success = False
            MAX_REFLECTION_RETRIES = 2 # Попробуем 2 раза, чтобы не зациклиться надолго
            for attempt in range(MAX_REFLECTION_RETRIES):
                print(f"   -> Попытка рефлексии {attempt + 1}/{MAX_REFLECTION_RETRIES}...")
                # Получаем текущий план ДО попытки обновления
                original_plan_json = json.dumps(world_model.get_full_context()['dynamic_knowledge']['strategic_plan'])

                updated_plan = strategist.reflect_and_update_plan(world_model)
                
                # Проверяем, действительно ли план изменился. Если нет, значит была ошибка.
                if updated_plan and json.dumps(updated_plan) != original_plan_json:
                    world_model.update_strategic_plan(updated_plan)
                    print("   -> Рефлексия успешна, план обновлен.")
                    reflection_success = True
                    break # Выходим из цикла попыток
                else:
                    print(f"   !!! Попытка рефлексии {attempt + 1} провалена (план не изменился). Пауза 15 секунд перед повтором.")
                    time.sleep(15)

            if not reflection_success:
                print("!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось провести рефлексию после нескольких попыток. Завершение работы.")
                # Здесь можно либо завершить работу, либо пометить проект как FAILED
                world_model.dynamic_knowledge['strategic_plan']['main_goal_status'] = 'FAILED'
                world_model._save_state_to_disk() # Сохраняем статус FAILED
                break # Выходим из главного цикла while True

            print(f"   [Оркестратор] Пауза после рефлексии на {STRATEGIST_COOLDOWN_SECONDS} секунд...")
            time.sleep(STRATEGIST_COOLDOWN_SECONDS)
            
            # Этот код остается таким же, но теперь он будет вызван только после УСПЕШНОЙ рефлексии
            if world_model.get_full_context()['dynamic_knowledge']['strategic_plan'].get("main_goal_status") == "READY_FOR_FINAL_BRIEF":
                print("--- Стратег решил, что информации достаточно. Переходим к написанию финального отчета. ---")
                break
            
            # --- НАЧАЛО: ПРОТОКОЛ АВАРИЙНОГО ПЕРЕХОДА ФАЗЫ ---
            if not world_model.get_active_tasks():
                print("!!! [Supervisor] ВНИМАНИЕ: Стратег не создал новых задач. Активирую протокол принудительного перехода.")
                
                current_plan = world_model.get_full_context()['dynamic_knowledge']['strategic_plan']
                phases = current_plan.get("phases", [])
                
                # 1. Находим и принудительно завершаем текущую фазу (если она еще активна)
                for i, phase in enumerate(phases):
                    if phase.get("status") == "IN_PROGRESS":
                        print(f"   [Supervisor] -> Принудительно завершаю фазу '{phase.get('phase_name')}'.")
                        phases[i]["status"] = "COMPLETED"
                        break
                
                # 2. Находим и активируем следующую ожидающую фазу
                next_phase_activated = False
                for i, phase in enumerate(phases):
                    if phase.get("status") == "PENDING":
                        print(f"   [Supervisor] -> Активирую следующую фазу '{phase.get('phase_name')}'.")
                        phases[i]["status"] = "IN_PROGRESS"
                        next_phase_activated = True
                        break
                
                # 3. Сохраняем измененный план
                world_model.update_strategic_plan(current_plan)

                if not next_phase_activated:
                    print("!!! [Supervisor] КРИТИЧЕСКАЯ ОШИБКА: Следующая фаза для активации не найдена. Вероятно, план выполнен или поврежден. Остановка.")
                    break # Выходим из главного цикла, так как работа действительно закончена
            # --- КОНЕЦ ПРОТОКОЛА ---
            
            continue

        task_to_run = active_tasks[0]
        task_id = task_to_run['task_id'] # Удобно иметь ID в переменной
        
        # --- ОБНОВЛЕННЫЙ БЛОК TRY...EXCEPT ---
        try:
            claims = expert_team.execute_task(task_to_run, world_model)
            
            if claims:
                world_model.update_task_status(task_id, 'COMPLETED')
                world_model.log_transaction({'task': task_to_run, 'results': claims})
            else:
                # Если claims пустые, но ошибки не было, значит, эксперт ничего не нашел или был конфликт
                # Статус задачи уже обновлен внутри execute_task, если нужно
                # Просто логируем, что результат пустой
                world_model.log_transaction({'task': task_to_run, 'results': "No new verified claims generated"})

        except SearchAPIFailureError as e:
            print(f"!!! ОРКЕСТРАТОР: Произошла ошибка поиска при выполнении задачи {task_id}. Ошибка: {e}")
            
            current_retries = task_to_run.get('retry_count', 0)
            if current_retries < MAX_RETRIES:
                world_model.increment_task_retry_count(task_id)
                print(f"   -> Попытка {current_retries + 1}/{MAX_RETRIES}. Повтор через {RETRY_DELAY_SECONDS} секунд...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                print(f"!!! ОРКЕСТРАТОР: Превышен лимит ({MAX_RETRIES}) повторных попыток для задачи {task_id}. Задача окончательно провалена.")
                world_model.update_task_status(task_id, 'FAILED')
                world_model.log_transaction({'task': task_to_run, 'results': f"CRITICAL SEARCH ERROR after {MAX_RETRIES} retries: {e}"})

        except ResourceExhausted as e:
            print(f"!!! СИСТЕМНЫЙ СБОЙ: ДОСТИГНУТ ДНЕВНОЙ ЛИМИТ API. ОСТАНАВЛИВАЮ ВСЮ СИСТЕМУ.")
            print(f"   -> Задача '{task_id}' остается в статусе PENDING.")
            print(f"   -> Ошибка: {e}")
            # Завершаем работу, так как дневной лимит исчерпан
            sys.exit(1)

        except Exception as e:
            print(f"!!! ОРКЕСТРАТОР: Произошла НЕПРЕДВИДЕННАЯ критическая ошибка при выполнении задачи {task_id}. Задача провалена. Ошибка: {e}")
            world_model.update_task_status(task_id, 'FAILED')
            world_model.log_transaction({'task': task_to_run, 'results': f"UNHANDLED CRITICAL ERROR: {e}"})
        print(f"   [Оркестратор] Пауза после задачи на {TASK_COOLDOWN_SECONDS} секунд...")
        time.sleep(TASK_COOLDOWN_SECONDS)

    # --- ФИНАЛЬНЫЙ ОТЧЕТ С ЦИКЛОМ ВАЛИДАЦИИ ---
    print("\n--- Создание и валидация финальных отчетов... ---")
    MAX_VALIDATION_RETRIES = 3

    # --- Генерация и валидация Executive Summary ---
    summary_content = ""
    try:
        for i in range(MAX_VALIDATION_RETRIES):
            print(f"\n   -> Попытка {i+1}/{MAX_VALIDATION_RETRIES} генерации Executive Summary...")
            feedback = summary_content # Используем предыдущий неудачный результат как фидбэк
            # Генерируем черновик
            draft_summary = strategist.write_executive_summary(world_model, feedback=feedback)
            # Валидируем черновик
            validation_report = strategist.validate_artifact(
            llms['source_auditor'], # <-- ИСПОЛЬЗУЕМ ДЕШЕВУЮ И ДОСТУПНУЮ МОДЕЛЬ
            draft_summary,
            required_sections=["Executive Summary", "Концепция Продукта", "Дорожная Карта"]
            )
            if validation_report.get("is_valid"):
                summary_content = draft_summary
                print("   -> Executive Summary успешно сгенерирован и прошел валидацию.")
                break
            else:
                summary_content = f"Validation failed. Reasons: {validation_report.get('reasons', [])}"
                print(f"   !!! Попытка {i+1} провалена. Отправляю на доработку...")
                time.sleep(10)
    except ResourceExhausted as e:
        print("!!! ОШИБКА БЮДЖЕТА: Не удалось сгенерировать Executive Summary из-за исчерпания лимита API.")
        summary_content = f"Validation failed. Reason: {e}"

    # Сохраняем результат (успешный или отчет об ошибке)
    summary_file_path = os.path.join(world_model.output_dir, "Executive_Summary_For_Director.md")
    if summary_content and "Validation failed" not in summary_content:
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
        print(f"-> Краткая аналитическая записка сохранена в {summary_file_path}")
    else:
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(f"# ГЕНЕРАЦИЯ ПРОВАЛЕНА\n\nНе удалось создать качественный документ после {MAX_VALIDATION_RETRIES} попыток.\n\nПоследняя ошибка: {summary_content}")
        print(f"!!! ОШИБКА: Не удалось сгенерировать Executive Summary. Отчет об ошибке сохранен в {summary_file_path}")

    # --- Генерация и валидация Extended Brief (аналогичный цикл) ---
    brief_content = ""
    try:
        for i in range(MAX_VALIDATION_RETRIES):
            print(f"\n   -> Попытка {i+1}/{MAX_VALIDATION_RETRIES} генерации Extended Brief...")
            feedback = brief_content
            draft_brief = strategist.write_extended_brief(world_model, feedback=feedback)
            validation_report = strategist.validate_artifact(
                llms['source_auditor'], # <-- ИСПОЛЬЗУЕМ ДЕШЕВУЮ И ДОСТУПНУЮ МОДЕЛЬ
                draft_brief,
                required_sections=["Анализ Активов ТГУ", "Конкурентный Ландшафт", "Бизнес-Кейс"]
            )
            if validation_report.get("is_valid"):
                brief_content = draft_brief
                print("   -> Extended Brief успешно сгенерирован и прошел валидацию.")
                break
            else:
                brief_content = f"Validation failed. Reasons: {validation_report.get('reasons', [])}"
                print(f"   !!! Попытка {i+1} провалена. Отправляю на доработку...")
                time.sleep(10)
    except ResourceExhausted as e:
        print("!!! ОШИБКА БЮДЖЕТА: Не удалось сгенерировать Extended Brief из-за исчерпания лимита API.")
        brief_content = f"Validation failed. Reason: {e}"

    brief_file_path = os.path.join(world_model.output_dir, "Extended_Brief_For_PO.md")
    if brief_content and "Validation failed" not in brief_content:
        with open(brief_file_path, "w", encoding="utf-8") as f:
            f.write(brief_content)
        print(f"-> Подробный аналитический обзор сохранен в {brief_file_path}")
    else:
        with open(brief_file_path, "w", encoding="utf-8") as f:
            f.write(f"# ГЕНЕРАЦИЯ ПРОВАЛЕНА\n\nНе удалось создать качественный документ после {MAX_VALIDATION_RETRIES} попыток.\n\nПоследняя ошибка: {brief_content}")
        print(f"!!! ОШИБКА: Не удалось сгенерировать Extended Brief. Отчет об ошибке сохранен в {brief_file_path}")
        
    print(f"\n--- РАБОТА УСПЕШНО ЗАВЕРШЕНА ---")

if __name__ == "__main__":
    main()