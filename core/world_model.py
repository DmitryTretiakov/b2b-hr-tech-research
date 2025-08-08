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
    def __init__(self, static_context: dict, budget_manager: APIBudgetManager, output_dir: str = "output", force_fresh_start: bool = False):
        self.static_context = static_context
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, "logs")
        self.cache_dir = os.path.join(output_dir, "cache")
        
        # Определяем пути для всех персистентных артефактов
        self.state_file_path = os.path.join(self.output_dir, "system_state.json")
        self.index_path = os.path.join(self.output_dir, "faiss.index")
        self.id_map_path = os.path.join(self.output_dir, "id_map.json")

        # Обрабатываем флаг --fresh-start, удаляя все старые данные
        if force_fresh_start:
            print("!!! [WorldModel] Активирован режим 'fresh-start'. Удаляю старые файлы состояния и индекса.")
            if os.path.exists(self.state_file_path): os.remove(self.state_file_path)
            if os.path.exists(self.index_path): os.remove(self.index_path)
            if os.path.exists(self.id_map_path): os.remove(self.id_map_path)
                
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
            "transaction_log": [],
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
        """Сохраняет полный объект dynamic_knowledge в JSON-файл."""
        print("   [WorldModel] -> Сохраняю текущее состояние на диск...")
        try:
            with open(self.state_file_path, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_knowledge, f, ensure_ascii=False, indent=2)
            print("   [WorldModel] <- Состояние успешно сохранено.")
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА [WorldModel]: Не удалось сохранить состояние в {self.state_file_path}. Ошибка: {e}")

    # --- НОВЫЙ ПРИВАТНЫЙ МЕТОД ДЛЯ ЗАГРУЗКИ ---
    def _load_state_from_disk(self):
        """Загружает состояние из JSON-файла, если он существует."""
        if os.path.exists(self.state_file_path):
            print(f"   [WorldModel] -> Найден файл состояния {self.state_file_path}. Загружаю...")
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    self.dynamic_knowledge = json.load(f)
                print("   [WorldModel] <- Состояние успешно загружено.")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"!!! ВНИМАНИЕ [WorldModel]: Не удалось загрузить состояние из файла. Будет использовано пустое состояние. Ошибка: {e}")
        else:
            print("   [WorldModel] Файл состояния не найден. Используется пустое состояние по умолчанию.")

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

    def log_transaction(self, transaction: dict):
        """Логирует транзакцию, сохраняет ее в отдельный файл и СОХРАНЯЕТ ОБЩЕЕ СОСТОЯНИЕ."""
        try:
            task_id = transaction.get('task', {}).get('task_id', 'unknown_task')
            assignee = transaction.get('task', {}).get('assignee', 'unknown_assignee')
            
            tx_id = (f"tx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                     f"{assignee}_{task_id}")
            
            transaction['transaction_id'] = tx_id
            self.dynamic_knowledge["transaction_log"].append(transaction)
            
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