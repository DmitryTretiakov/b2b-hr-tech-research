# core/world_model.py
import json
import os
from datetime import datetime
from core.semantic_index import SemanticIndex
from core.budget_manager import APIBudgetManager
from core.embedding_client import GeminiEmbeddingClient

class WorldModel:
    def __init__(self, static_context: dict, budget_manager: APIBudgetManager, output_dir: str = "output", force_fresh_start: bool = False, reset_plan_only: bool = False):
        self.static_context = static_context
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, "logs")
        self.cache_dir = os.path.join(output_dir, "cache")
        
        self.state_file_path = os.path.join(self.output_dir, "system_state.json")
        self.index_path = os.path.join(self.output_dir, "faiss.index")
        self.id_map_path = os.path.join(self.output_dir, "id_map.json")

        self._cleanup_temp_files()

        if force_fresh_start:
            print("!!! [WorldModel] Активирован режим 'fresh-start'. Удаляю старые файлы.")
            if os.path.exists(self.state_file_path): os.remove(self.state_file_path)
            if os.path.exists(self.index_path): os.remove(self.index_path)
            if os.path.exists(self.id_map_path): os.remove(self.id_map_path)
        
        elif reset_plan_only:
            print("!!! [WorldModel] Активирован режим 'reset-plan-only'.")
            if os.path.exists(self.state_file_path):
                try:
                    with open(self.state_file_path, 'r', encoding='utf-8') as f:
                        temp_state = json.load(f)
                    temp_state['strategic_plan'] = {}
                    with open(self.state_file_path, 'w', encoding='utf-8') as f:
                        json.dump(temp_state, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print(f"!!! [WorldModel] ОШИБКА при сбросе плана: {e}.")
                
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        embedding_client = GeminiEmbeddingClient(budget_manager=budget_manager)
        self.semantic_index = SemanticIndex(embedding_client=embedding_client, budget_manager=budget_manager)

        self.dynamic_knowledge = {"strategic_plan": {}, "knowledge_base": {}, "generated_artifacts": {}}
        
        index_loaded = self.semantic_index.load_from_disk(self.index_path, self.id_map_path)
        self._load_state_from_disk()

        if self.dynamic_knowledge.get('knowledge_base') and not index_loaded:
            print("!!! [WorldModel] Обнаружена База Знаний, но отсутствует индекс. Запускаю перестройку...")
            self.semantic_index.rebuild_from_kb(self.dynamic_knowledge['knowledge_base'], self.index_path, self.id_map_path)

        print("-> WorldModel инициализирован.")

    def _load_state_from_disk(self):
        if os.path.exists(self.state_file_path):
            try:
                with open(self.state_file_path, "r", encoding="utf-8") as f:
                    self.dynamic_knowledge.update(json.load(f))
                print("   [WorldModel] Состояние успешно загружено с диска.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"!!! [WorldModel] ВНИМАНИЕ: Не удалось загрузить файл состояния. Ошибка: {e}")

    def save_state(self):
        """
        Атомарно сохраняет полное состояние системы (state + index).
        Этот метод теперь является ЕДИНСТВЕННОЙ точкой сохранения.
        """
        print("   [WorldModel] -> Начало атомарной транзакции сохранения...")
        tmp_state_path = self.state_file_path + ".tmp"
        tmp_files_to_commit = []

        try:
            # Фаза 1: Запись во временные файлы
            tmp_index_path, tmp_id_map_path = self.semantic_index.save_to_disk(self.index_path, self.id_map_path)
            tmp_files_to_commit.extend([tmp_index_path, tmp_id_map_path])

            with open(tmp_state_path, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_knowledge, f, ensure_ascii=False, indent=2)
            tmp_files_to_commit.append(tmp_state_path)
            
            # Фаза 2: Атомарное переименование (коммит)
            print("   [WorldModel] -> Коммит транзакции...")
            os.replace(tmp_index_path, self.index_path)
            os.replace(tmp_id_map_path, self.id_map_path)
            os.replace(tmp_state_path, self.state_file_path)
            
            print("   [WorldModel] <- Транзакция успешно завершена.")

        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА [WorldModel]: Транзакция сохранения провалена. Ошибка: {e}. Выполняю откат...")
            # Откат не требуется, так как оригинальные файлы не были затронуты
            raise e
        finally:
            # Очистка временных файлов в любом случае
            for tmp_path in tmp_files_to_commit:
                if os.path.exists(tmp_path) and ".tmp" in tmp_path:
                    os.remove(tmp_path)

    def update_strategic_plan(self, plan: dict):
        """Обновляет план в памяти."""
        if plan and isinstance(plan, dict) and "phases" in plan:
            self.dynamic_knowledge["strategic_plan"] = plan
            print("   [WorldModel] Стратегический план обновлен (в памяти).")
        else:
            print("!!! [WorldModel] ОШИБКА: Попытка обновить план невалидными данными.")

    def add_claims_to_kb(self, claims: list):
        """Добавляет/обновляет утверждения и их векторы в памяти."""
        if not claims: return
        added_count = 0
        for claim in claims:
            if isinstance(claim, dict) and 'claim_id' in claim:
                claim_id = claim['claim_id']
                is_new = claim_id not in self.dynamic_knowledge["knowledge_base"]
                self.dynamic_knowledge["knowledge_base"][claim_id] = claim
                if is_new:
                    self.semantic_index.add_claim(claim_id, claim['statement'])
                added_count += 1
        if added_count > 0:
            print(f"   [WorldModel] Добавлено/обновлено {added_count} утверждений в Базе Знаний (в памяти).")

    def update_task_status(self, task_id: str, new_status: str):
        """Обновляет статус задачи в памяти."""
        plan = self.dynamic_knowledge.get("strategic_plan", {})
        for phase in plan.get("phases", []):
            for task in phase.get("tasks", []):
                if task.get("task_id") == task_id:
                    task["status"] = new_status
                    return
        print(f"   [WorldModel] !!! Внимание: Задача с ID '{task_id}' не найдена в плане.")
        
    def log_transaction(self, transaction: dict):
        """Записывает лог транзакции в файл."""
        try:
            task_id = transaction.get('task', {}).get('task_id', 'unknown_task')
            tx_id = f"tx_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{task_id}"
            transaction['transaction_id'] = tx_id
            with open(os.path.join(self.log_dir, f"{tx_id}.json"), "w", encoding="utf-8") as f:
                json.dump(transaction, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"!!! ОШИБКА: Не удалось сохранить лог транзакции. Ошибка: {e}")

    # ... Остальные методы (get_active_tasks, get_full_context и т.д.) остаются без изменений ...
    def _cleanup_temp_files(self):
        """Удаляет все временные файлы сохранения, оставшиеся от предыдущих сбоев."""
        print("   [WorldModel] Проверка на наличие временных файлов...")
        for path in [self.state_file_path, self.index_path, self.id_map_path]:
            tmp_path = path + ".tmp"
            if os.path.exists(tmp_path):
                print(f"      -> Найден и удален временный файл: {tmp_path}")
                os.remove(tmp_path)

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

    def get_full_context(self) -> dict:
        """Возвращает полный слепок текущего состояния для агентов."""
        return {
            "static_context": self.static_context,
            "dynamic_knowledge": self.dynamic_knowledge
        }