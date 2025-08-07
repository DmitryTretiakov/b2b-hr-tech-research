# core/world_model.py
import json
import os
from datetime import datetime

# Убираем импорт sanitize_filename, так как он больше не используется в этом файле
# from utils.helpers import sanitize_filename 

class WorldModel:
    """
    Центральное, персистентное хранилище состояния и знаний для всей системы.
    Управляет планом, базой знаний и логами.
    Сохраняет свое состояние на диск при каждом изменении.
    """
    def __init__(self, static_context: dict, output_dir: str = "output", force_fresh_start: bool = False):
        self.static_context = static_context
        self.output_dir = output_dir
        self.kb_dir = os.path.join(output_dir, "knowledge_base")
        self.log_dir = os.path.join(output_dir, "logs")
        self.cache_dir = os.path.join(output_dir, "cache")
        
        # --- НОВЫЙ АТРИБУТ ДЛЯ ФАЙЛА СОСТОЯНИЯ ---
        self.state_file_path = os.path.join(self.output_dir, "system_state.json")
        if force_fresh_start and os.path.exists(self.state_file_path):
            print(f"!!! [WorldModel] Активирован режим 'fresh-start'. Удаляю старый файл состояния: {self.state_file_path}")
            try:
                os.remove(self.state_file_path)
            except OSError as e:
                print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось удалить файл состояния. Ошибка: {e}")
                
        # Создаем все директории, если их нет
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        # Инициализируем пустое состояние по умолчанию
        self.dynamic_knowledge = {
            "strategic_plan": {},
            "knowledge_base": {},
            "transaction_log": []
        }
        
        # --- НОВЫЙ ВЫЗОВ: ПЫТАЕМСЯ ЗАГРУЗИТЬ СОСТОЯНИЕ ПРИ СТАРТЕ ---
        self._load_state_from_disk()

        print("-> WorldModel инициализирован (состояние загружено, если найдено).")
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
        Добавляет или ОБНОВЛЯЕТ список "Утверждений" в базе знаний и СОХРАНЯЕТ СОСТОЯНИЕ.
        Теперь может обрабатывать и одиночные утверждения.
        """
        if not isinstance(claims, list):
            claims = [claims] # Оборачиваем одиночный claim в список

        if not claims:
            return
        
        added_count = 0
        for claim in claims:
            if isinstance(claim, dict) and 'claim_id' in claim:
                self.dynamic_knowledge["knowledge_base"][claim['claim_id']] = claim
                added_count += 1
        
        if added_count > 0:
            print(f"   [WorldModel] Добавлено/обновлено {added_count} утверждений в Базе Знаний.")
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