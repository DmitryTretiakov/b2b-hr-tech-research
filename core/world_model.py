# core/world_model.py
import json
import os
from datetime import datetime
from utils.helpers import sanitize_filename

class WorldModel:
    """
    Центральное хранилище состояния и знаний для всей системы.
    Управляет планом, базой знаний и логами.
    """
    def __init__(self, static_context: dict, output_dir: str = "output"):
        self.static_context = static_context
        self.output_dir = output_dir
        self.kb_dir = os.path.join(output_dir, "knowledge_base")
        self.log_dir = os.path.join(output_dir, "logs")
        self.cache_dir = os.path.join(output_dir, "cache")
        
        # Создаем все директории, если их нет
        os.makedirs(self.kb_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.dynamic_knowledge = {
            "strategic_plan": {},
            "knowledge_base": {}, # Словарь для хранения "Утверждений" (Claims)
            "transaction_log": []
        }
        print("-> WorldModel инициализирован.")

    def update_strategic_plan(self, plan: dict):
        """Обновляет или заменяет стратегический план."""
        if plan and isinstance(plan, dict):
            self.dynamic_knowledge["strategic_plan"] = plan
            print("   [WorldModel] Стратегический план обновлен.")
        else:
            print("!!! [WorldModel] ОШИБКА: Попытка обновить план невалидными данными.")

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
        """Добавляет список "Утверждений" в базу знаний."""
        if not claims or not isinstance(claims, list):
            return
        
        added_count = 0
        for claim in claims:
            if isinstance(claim, dict) and 'claim_id' in claim:
                self.dynamic_knowledge["knowledge_base"][claim['claim_id']] = claim
                added_count += 1
        
        if added_count > 0:
            print(f"   [WorldModel] Добавлено {added_count} утверждений в Базу Знаний.")

    def update_task_status(self, task_id: str, new_status: str):
        """Находит задачу по ID во всем плане и обновляет ее статус."""
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
        if not task_found:
            print(f"   [WorldModel] !!! Внимание: Задача с ID '{task_id}' не найдена в плане.")

    def log_transaction(self, transaction: dict):
        """Логирует транзакцию и сохраняет ее в отдельный файл."""
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
        except Exception as e:
            print(f"!!! ОШИБКА: Не удалось сохранить лог транзакции {tx_id}. Ошибка: {e}")
        
    def get_full_context(self) -> dict:
        """Возвращает полный слепок текущего состояния для агентов."""
        return {
            "static_context": self.static_context,
            "dynamic_knowledge": self.dynamic_knowledge
        }
    def increment_task_retry_count(self, task_id: str):
        """Находит задачу и увеличивает ее счетчик попыток."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        task_found = False
        for phase in plan.get("phases", []):
            for task in phase.get("tasks", []):
                if task.get("task_id") == task_id:
                    # Убедимся, что счетчик существует, и увеличим его
                    current_retries = task.get('retry_count', 0)
                    task['retry_count'] = current_retries + 1
                    print(f"   [WorldModel] Счетчик попыток для задачи '{task_id}' увеличен до {task['retry_count']}.")
                    task_found = True
                    break
            if task_found:
                break
        if not task_found:
            print(f"   [WorldModel] !!! Внимание: Задача с ID '{task_id}' не найдена для инкремента попыток.")