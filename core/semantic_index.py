# core/semantic_index.py
import faiss
import numpy as np
import json
import os
import traceback
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.budget_manager import APIBudgetManager
from google.api_core.exceptions import ResourceExhausted

class SemanticIndex:
    def __init__(self, embedding_model: GoogleGenerativeAIEmbeddings, budget_manager: APIBudgetManager, save_every_n: int = 10):
        self.embedding_model = embedding_model
        self.budget_manager = budget_manager
        
        # --- ДИНАМИЧЕСКОЕ ОПРЕДЕЛЕНИЕ РАЗМЕРНОСТИ ---
        try:
            print("   [SemanticIndex] Определяю размерность векторов модели эмбеддингов...")
            self.dimension = self._get_embedding_dimension()
            print(f"   [SemanticIndex] Размерность определена: {self.dimension}")
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось определить размерность модели. Использую значение по умолчанию 768. Ошибка: {e}")
            self.dimension = 768 # Отказоустойчивое значение по умолчанию
        # --- КОНЕЦ ДИНАМИЧЕСКОГО ОПРЕДЕЛЕНИЯ ---

        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = []
        self.save_every_n = save_every_n
        self._add_counter = 0
        print(f"-> SemanticIndex (FAISS) инициализирован с динамически определенной размерностью {self.dimension}.")

    def _get_embedding_dimension(self) -> int:
        """Делает один тестовый запрос для определения размерности векторов."""
        model_name = self.embedding_model.model
        if not self.budget_manager.can_i_spend(model_name):
            raise ResourceExhausted("Невозможно определить размерность, так как лимит API эмбеддингов исчерпан.")
        
        # Используем простой текст, который гарантированно не вызовет проблем с безопасностью
        test_vector = self.embedding_model.embed_query("test")
        self.budget_manager.record_spend(model_name) # Мы потратили один вызов, его нужно учесть
        return len(test_vector)

    def save_to_disk(self, index_path: str, id_map_path: str):
        """Сохраняет индекс FAISS и карту ID на диск."""
        print(f"   [SemanticIndex] -> Сохраняю индекс ({self.index.ntotal} векторов) на диск...")
        try:
            faiss.write_index(self.index, index_path)
            with open(id_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.id_map, f)
            print("   [SemanticIndex] <- Индекс и карта ID успешно сохранены.")
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА [SemanticIndex]: Не удалось сохранить индекс. Ошибка: {e}")

    def load_from_disk(self, index_path: str, id_map_path: str) -> bool:
        """Загружает индекс FAISS и карту ID с диска."""
        if os.path.exists(index_path) and os.path.exists(id_map_path):
            print("   [SemanticIndex] -> Загружаю индекс с диска...")
            try:
                self.index = faiss.read_index(index_path)
                # ПРОВЕРКА СОВМЕСТИМОСТИ
                if self.index.d != self.dimension:
                    print(f"!!! ВНИМАНИЕ [SemanticIndex]: Размерность загруженного индекса ({self.index.d}) не совпадает с размерностью текущей модели ({self.dimension}). Индекс будет проигнорирован.")
                    return False
                with open(id_map_path, 'r', encoding='utf-8') as f:
                    self.id_map = json.load(f)
                print(f"   [SemanticIndex] <- Индекс успешно загружен ({self.index.ntotal} векторов).")
                return True
            except Exception as e:
                print(f"!!! ВНИМАНИЕ [SemanticIndex]: Не удалось загрузить индекс. Будет создан новый. Ошибка: {e}")
                return False
        return False

    def add_claim(self, claim_id: str, claim_text: str, index_path: str, id_map_path: str):
        """Добавляет одно утверждение в индекс."""
        if claim_id in self.id_map:
            return
        try:
            model_name = self.embedding_model.model
            if not self.budget_manager.can_i_spend(model_name):
                raise ResourceExhausted(f"Daily budget for embedding model {model_name} reached.")
            vector = self.embedding_model.embed_query(claim_text)
            self.budget_manager.record_spend(model_name)
            vector_np = np.array([vector], dtype=np.float32)
            self.index.add(vector_np)
            self.id_map.append(claim_id)
            self._add_counter += 1
            if self._add_counter % self.save_every_n == 0:
                self.save_to_disk(index_path, id_map_path)
        except Exception as e:
            print(f"!!! ОШИБКА [SemanticIndex]: Не удалось добавить claim '{claim_id}'.")
            print(f"   -> Тип ошибки: {type(e).__name__}, Сообщение: {e}")
            print(f"   -> Traceback:\n{traceback.format_exc()}")
            raise e

    def find_similar_claim_ids(self, query_text: str, top_k: int = 5) -> list[str]:
        """Ищет похожие утверждения в индексе."""
        if self.index.ntotal == 0:
            return []
        try:
            model_name = self.embedding_model.model
            if not self.budget_manager.can_i_spend(model_name):
                raise ResourceExhausted(f"Daily budget for embedding model {model_name} reached.")
            query_vector = self.embedding_model.embed_query(query_text)
            self.budget_manager.record_spend(model_name)
            query_vector_np = np.array([query_vector], dtype=np.float32)
            distances, indices = self.index.search(query_vector_np, k=min(top_k, self.index.ntotal))
            similar_ids = [self.id_map[i] for i in indices[0]]
            return similar_ids
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА [SemanticIndex]: Не удалось выполнить поиск.")
            print(f"   -> Тип ошибки: {type(e).__name__}, Сообщение: {e}")
            print(f"   -> Traceback:\n{traceback.format_exc()}")
            raise e

    def rebuild_from_kb(self, knowledge_base: dict, index_path: str, id_map_path: str):
        """АВАРИЙНЫЙ МЕТОД: Полностью перестраивает индекс из Базы Знаний."""
        print("   [SemanticIndex] !!! ВНИМАНИЕ: Запущена полная перестройка индекса из Базы Знаний...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = []
        for claim_id, claim_data in knowledge_base.items():
            self._add_claim_internal(claim_id, claim_data['statement'])
        print(f"   [SemanticIndex] <- Индекс перестроен. Всего векторов: {self.index.ntotal}.")
        self.save_to_disk(index_path, id_map_path)

    def _add_claim_internal(self, claim_id: str, claim_text: str):
        """Внутренний метод для добавления без сохранения и с пропуском ошибок."""
        try:
            model_name = self.embedding_model.model
            if not self.budget_manager.can_i_spend(model_name):
                print(f"!!! [Бюджет] Пропуск claim '{claim_id}' при перестройке из-за лимита API.")
                return
            vector = self.embedding_model.embed_query(claim_text)
            self.budget_manager.record_spend(model_name)
            vector_np = np.array([vector], dtype=np.float32)
            self.index.add(vector_np)
            self.id_map.append(claim_id)
        except Exception as e:
            print(f"!!! ОШИБКА [SemanticIndex]: Пропускаю claim '{claim_id}' при перестройке.")
            print(f"   -> Тип ошибки: {type(e).__name__}, Сообщение: {e}")