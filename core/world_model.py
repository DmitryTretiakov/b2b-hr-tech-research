# core/world_model.py
import json
import os
from datetime import datetime
from core.semantic_index import SemanticIndex
from core.budget_manager import APIBudgetManager
from core.embedding_client import GeminiEmbeddingClient

class WorldModel:
    """
    Центральное, персистентное хранилище состояния и знаний для всей системы.
    Управляет планом, базой знаний и логами.
    Сохраняет свое состояние на диск при каждом изменении.
    """
    def __init__(self, static_context: dict, budget_manager: APIBudgetManager, output_dir: str = "output", force_fresh_start: bool = False, reset_plan_only: bool = False):
        self.static_context = static_context
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, "logs")
        self.cache_dir = os.path.join(output_dir, "cache")
        
        # Определяем пути для всех персистентных артефактов
        self.state_file_path = os.path.join(self.output_dir, "system_state.json")
        self.index_path = os.path.join(self.output_dir, "faiss.index")
        self.id_map_path = os.path.join(self.output_dir, "id_map.json")

        self._cleanup_temp_files()

        # Обрабатываем флаг --fresh-start, удаляя все старые данные
        if force_fresh_start:
            print("!!! [WorldModel] Активирован режим 'fresh-start'. Удаляю старые файлы состояния и индекса.")
            if os.path.exists(self.state_file_path): os.remove(self.state_file_path)
            if os.path.exists(self.index_path): os.remove(self.index_path)
            if os.path.exists(self.id_map_path): os.remove(self.id_map_path)


        elif reset_plan_only:
           print("!!! [WorldModel] Активирован режим 'reset-plan-only'. Сбрасываю план, но сохраняю Базу Знаний.")
           if os.path.exists(self.state_file_path):
               try:
                   with open(self.state_file_path, 'r', encoding='utf-8') as f:
                       temp_state = json.load(f)
                   temp_state['strategic_plan'] = {} # Сбрасываем только план
                   with open(self.state_file_path, 'w', encoding='utf-8') as f:
                       json.dump(temp_state, f, ensure_ascii=False, indent=2)
                   print("   [WorldModel] -> Стратегический план успешно сброшен.")
               except Exception as e:
                   print(f"!!! [WorldModel] ОШИБКА при сбросе плана: {e}. Запускаюсь как обычно.") 
        # Создаем все директории, если их нет
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        embedding_client = GeminiEmbeddingClient(budget_manager=budget_manager)
        
        self.semantic_index = SemanticIndex(
            embedding_client=embedding_client, # <-- ПЕРЕДАЕМ НАШ КЛИЕНТ
            budget_manager=budget_manager
        )

        # Инициализируем пустое состояние по умолчанию
        self.dynamic_knowledge = {
            "strategic_plan": {},
            "knowledge_base": {},
            "generated_artifacts": {}
        }
        
        # --- Отказоустойчивая логика загрузки ---
        # 1. Сначала пытаемся загрузить персистентный индекс с диска
        index_loaded = self.semantic_index.load_from_disk(self.index_path, self.id_map_path)
        
        # 2. Затем пытаемся загрузить основное состояние (план, KB)
        self._load_state_from_disk()

        # 3. Сверяем состояние. Если База Знаний загрузилась, а индекс - нет,
        #    запускаем аварийную перестройку индекса.
        if self.dynamic_knowledge.get('knowledge_base') and not index_loaded:
            print("!!! [WorldModel] Обнаружена База Знаний, но отсутствует семантический индекс. Запускаю перестройку...")
            self.semantic_index.rebuild_from_kb(
                self.dynamic_knowledge['knowledge_base'], 
                self.index_path, 
                self.id_map_path
            )

        print("-> WorldModel инициализирован (состояние и индекс загружены, если найдены).")

    def add_task_to_plan(self, task: dict, phase_name: str = "Phase 1: Глубокая Разведка Активов ТГУ"):
        """
        Добавляет новую задачу в указанную или активную фазу плана.
        """
        print(f"   [WorldModel] -> Добавляю новую задачу '{task.get('task_id')}' в план...")
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        phase_found = False
        # Ищем фазу по имени
        for p in plan.get("phases", []):
            if p.get("phase_name") == phase_name:
                p.get("tasks", []).insert(0, task) # Вставляем в начало, чтобы повысить приоритет
                phase_found = True
                break
        
        # Если фаза по имени не найдена, ищем активную
        if not phase_found:
            for p in plan.get("phases", []):
                if p.get("status") == "IN_PROGRESS":
                    p.get("tasks", []).insert(0, task)
                    phase_found = True
                    break

        if phase_found:
            print(f"   [WorldModel] <- Задача '{task.get('task_id')}' добавлена.")
            self._save_state_to_disk()
        else:
            print(f"!!! [WorldModel] ОШИБКА: Не найдено подходящей фазы для добавления задачи.")
    
    # --- НОВЫЙ ПРИВАТНЫЙ МЕТОД ДЛЯ СОХРАНЕНИЯ ---
    def _save_state_to_disk(self):
        """
        Атомарно сохраняет полное состояние системы (state index) через
        двухфазную запись во временные файлы. Гарантирует консистентность.
        """
        print("   [WorldModel] -> Начало атомарной транзакции сохранения...")
        
        # Определяем пути к временным файлам
        tmp_state_path = self.state_file_path + ".tmp"
        tmp_files = [tmp_state_path]

        try:
            # --- ФАЗА 1: ПОДГОТОВКА И ЗАПИСЬ ВО ВРЕМЕННЫЕ ФАЙЛЫ ---

            # 1. Готовим и записываем индекс
            tmp_index_path, tmp_id_map_path = self.semantic_index.save_to_disk(self.index_path, self.id_map_path)
            tmp_files.extend([tmp_index_path, tmp_id_map_path])

            # 2. Готовим и записываем основное состояние
            with open(tmp_state_path, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_knowledge, f, ensure_ascii=False, indent=2)
            
            # --- ФАЗА 2: АТОМАРНЫЙ КОММИТ (ПЕРЕИМЕНОВАНИЕ) ---
            # Эта фаза выполняется только если в Фазе 1 не было исключений
            
            print("   [WorldModel] -> Коммит транзакции...")
            os.rename(tmp_index_path, self.index_path)
            os.rename(tmp_id_map_path, self.id_map_path)
            os.rename(tmp_state_path, self.state_file_path)
            
            print("   [WorldModel] <- Транзакция успешно завершена. Состояние сохранено.")

        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА [WorldModel]: Транзакция сохранения провалена. Ошибка: {e}. Выполняю откат...")
            # Откат не нужен, так как finally блок сделает всю работу по очистке
            raise e # Перевыбрасываем ошибку, чтобы система знала о сбое
        
        finally:
            # --- ОЧИСТКА ---
            # Этот блок выполнится ВСЕГДА, даже если была ошибка.
            # Он удалит все временные файлы, которые не были успешно переименованы.
            for tmp_path in tmp_files:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def _cleanup_temp_files(self):
        """Удаляет все временные файлы сохранения, оставшиеся от предыдущих сбоев."""
        print("   [WorldModel] Проверка на наличие временных файлов от предыдущих сессий...")
        for path in [self.state_file_path, self.index_path, self.id_map_path]:
            tmp_path = path + ".tmp"
            if os.path.exists(tmp_path):
                print(f"      -> Найден и удален временный файл: {tmp_path}")
                os.remove(tmp_path)

    def update_strategic_plan(self, plan: dict):
        """Обновляет или заменяет стратегический план и СОХРАНЯЕТ СОСТОЯНИЕ."""
        if plan and isinstance(plan, dict) and "phases" in plan:
            self.dynamic_knowledge["strategic_plan"] = plan
            print("   [WorldModel] Стратегический план обновлен.")
            self._save_state_to_disk() # <-- Сохранение
        else:
            print("!!! [WorldModel] ОШИБКА: Попытка обновить план невалидными данными. Состояние не сохранено.")

    def get_active_tasks(self) -> list:
        """Возвращает список всех задач со статусом PENDING из активной фазы."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        active_tasks = []
        for phase in plan.get("phases", []):
            if phase.get("status") == "IN_PROGRESS":
                for task in phase.get("tasks", []):
                    if task.get("status") == "PENDING":
                        active_tasks.append(task)
        return active_tasks

    def add_claims_to_kb(self, claims: list):
        """
        Добавляет или обновляет "Утверждения" в базе знаний, обновляет семантический индекс
        и сохраняет общее состояние на диск.
        """
        if not isinstance(claims, list):
            claims = [claims] # Позволяет работать с одиночными утверждениями

        if not claims:
            return
        
        added_count = 0
        for claim in claims:
            if isinstance(claim, dict) and 'claim_id' in claim:
                claim_id = claim['claim_id']
                is_new = claim_id not in self.dynamic_knowledge["knowledge_base"]
                
                # Обновляем/добавляем утверждение в Базу Знаний
                self.dynamic_knowledge["knowledge_base"][claim_id] = claim
                
                # Если утверждение новое, добавляем его в семантический индекс.
                # Метод add_claim внутри себя处理 инкрементальное сохранение индекса.
                if is_new:
                    self.semantic_index.add_claim(
                        claim_id, 
                        claim['statement'],
                        self.index_path,
                        self.id_map_path
                    )
                added_count += 1
        
        if added_count > 0:
            print(f"   [WorldModel] Добавлено/обновлено {added_count} утверждений в Базе Знаний.")
            # Сохраняем основной файл состояния (plan, kb, log)
            self._save_state_to_disk()

    def update_task_status(self, task_id: str, new_status: str):
        """Находит задачу по ID, обновляет ее статус и СОХРАНЯЕТ СОСТОЯНИЕ."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        task_found = False
        for phase in plan.get("phases", []):
            for task in phase.get("tasks", []):
                if task.get("task_id") == task_id:
                    task["status"] = new_status
                    print(f"   [WorldModel] Статус задачи '{task_id}' обновлен на '{new_status}'.")
                    task_found = True
                    break
            if task_found:
                break
        
        if task_found:
            self._save_state_to_disk() # <-- Сохранение
        else:
            print(f"   [WorldModel] !!! Внимание: Задача с ID '{task_id}' не найдена в плане.")

    def increment_task_retry_count(self, task_id: str):
        """Находит задачу, увеличивает ее счетчик попыток и СОХРАНЯЕТ СОСТОЯНИЕ."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        task_found = False
        for phase in plan.get("phases", []):
            for task in phase.get("tasks", []):
                if task.get("task_id") == task_id:
                    current_retries = task.get('retry_count', 0)
                    task['retry_count'] = current_retries + 1
                    print(f"   [WorldModel] Счетчик попыток для задачи '{task_id}' увеличен до {task['retry_count']}.")
                    task_found = True
                    break
            if task_found:
                break
        
        if task_found:
            self._save_state_to_disk() # <-- Сохранение
        else:
            print(f"   [WorldModel] !!! Внимание: Задача с ID '{task_id}' не найдена для инкремента попыток.")

    def get_reflection_context(self) -> dict:
        """
        Возвращает специальный, "облегченный" контекст для рефлексии Стратега.
        Не содержит тяжелых логов, но включает краткую сводку по последней фазе.
        """
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        last_phase_summary = "No phases completed yet."
        
        # Находим последнюю завершенную фазу
        completed_phases = [p for p in plan.get("phases", []) if p.get("status") == "COMPLETED"]
        if completed_phases:
            last_phase = completed_phases[-1]
            tasks = last_phase.get("tasks", [])
            completed_count = sum(1 for t in tasks if t.get("status") == "COMPLETED")
            failed_count = sum(1 for t in tasks if t.get("status") == "FAILED")
            last_phase_summary = (f"Summary for last phase '{last_phase.get('phase_name')}': "
                                f"{len(tasks)} total tasks, "
                                f"{completed_count} completed, {failed_count} failed.")

        return {
            "static_context": self.static_context,
            "dynamic_knowledge": {
                "strategic_plan": self.dynamic_knowledge.get("strategic_plan"),
                "knowledge_base": self.dynamic_knowledge.get("knowledge_base"),
                "last_phase_summary": last_phase_summary # <-- Краткая сводка вместо логов
            }
        }

    def log_transaction(self, transaction: dict):
        """Логирует транзакцию, сохраняет ее в отдельный файл и СОХРАНЯЕТ ОБЩЕЕ СОСТОЯНИЕ."""
        try:
            task_id = transaction.get('task', {}).get('task_id', 'unknown_task')
            assignee = transaction.get('task', {}).get('assignee', 'unknown_assignee')
            
            tx_id = (f"tx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                     f"{assignee}_{task_id}")
            
            transaction['transaction_id'] = tx_id            
            log_filename = f"{tx_id}.json"
            with open(os.path.join(self.log_dir, log_filename), "w", encoding="utf-8") as f:
                json.dump(transaction, f, ensure_ascii=False, indent=2)
            print(f"   [WorldModel] Транзакция {tx_id} залогирована.")
            self._save_state_to_disk() # <-- Сохранение
        except Exception as e:
            print(f"!!! ОШИБКА: Не удалось сохранить лог транзакции {tx_id}. Ошибка: {e}")
        
    def get_full_context(self) -> dict:
        """Возвращает полный слепок текущего состояния для агентов."""
        return {
            "static_context": self.static_context,
            "dynamic_knowledge": self.dynamic_knowledge
        }
    def save_artifact(self, artifact_name: str, artifact_content: str):
        """Сохраняет сгенерированный артефакт в состояние и на диск."""
        print(f"   [WorldModel] -> Сохраняю новый артефакт: {artifact_name}")
        self.dynamic_knowledge['generated_artifacts'][artifact_name] = artifact_content
        
        # Сохраняем и в общем файле состояния
        self._save_state_to_disk()
        
        # И как отдельный файл в output для удобства
        try:
            artifact_path = os.path.join(self.output_dir, artifact_name)
            with open(artifact_path, "w", encoding="utf-8") as f:
                f.write(artifact_content)
            print(f"   [WorldModel] <- Артефакт также сохранен в файл {artifact_path}")
        except Exception as e:
            print(f"!!! ОШИБКА: Не удалось сохранить артефакт в отдельный файл. Ошибка: {e}")

    def get_last_completed_phase_name(self) -> str:
        """Находит имя последней по порядку фазы со статусом COMPLETED."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        completed_phases = [p for p in plan.get("phases", []) if p.get("status") == "COMPLETED"]
        if completed_phases:
            # Возвращаем имя последней в списке завершенных фаз
            return completed_phases[-1].get("phase_name", "unknown_phase")
        return "initial_phase" # Если ни одна не завершена

    def has_task_for_assignee_in_phase(self, phase_name: str, assignee: str) -> bool:
        """Проверяет, есть ли в указанной фазе хотя бы одна задача для данного исполнителя."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        for phase in plan.get("phases", []):
            if phase.get("phase_name") == phase_name:
                for task in phase.get("tasks", []):
                    if task.get("assignee") == assignee:
                        return True
        return False
    
    def update_main_goal_status(self, new_status: str):
        """Обновляет только статус главной цели в плане."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        if plan:
            plan["main_goal_status"] = new_status
            print(f"   [WorldModel] Статус главной цели обновлен на '{new_status}'.")
            self._save_state_to_disk()