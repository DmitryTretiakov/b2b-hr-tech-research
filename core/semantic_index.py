# core/semantic_index.py
import faiss
import numpy as np
import json
import os
import traceback
import google.generativeai as genai
from core.budget_manager import APIBudgetManager
from google.api_core.exceptions import ResourceExhausted
from core.embedding_client import GeminiEmbeddingClient
from typing import Tuple

class SemanticIndex:
    def __init__(self, embedding_client: GeminiEmbeddingClient, budget_manager: APIBudgetManager, save_every_n: int = 10):
        self.embedding_client = embedding_client
        self.budget_manager = budget_manager
        
        try:
            print("   [SemanticIndex] Определяю размерность векторов...")
            self.dimension = self.embedding_client.get_embedding_dimension()
            print(f"   [SemanticIndex] Размерность определена: {self.dimension}")
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось определить размерность модели. Использую значение по умолчанию 768. Ошибка: {e}")
            self.dimension = 768
        
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = []
        self.save_every_n = save_every_n
        self._add_counter = 0
        print(f"-> SemanticIndex (FAISS) инициализирован с динамически определенной размерностью {self.dimension}.")

    def save_to_disk(self, index_path: str, id_map_path: str) -> Tuple[str, str]:
        """
        Выполняет "подготовительную" фазу сохранения: записывает индекс и карту ID
        во временные файлы и возвращает их пути.
        Не выполняет финальное переименование.
        """
        tmp_index_path = index_path + ".tmp"
        tmp_id_map_path = id_map_path + ".tmp"
        
        print(f"   [SemanticIndex] -> Подготовка к сохранению индекса во временные файлы...")
        try:
            faiss.write_index(self.index, tmp_index_path)
            with open(tmp_id_map_path, 'w', encoding='utf-8') as f:
                json.dump(self.id_map, f)
            print(f"   [SemanticIndex] <- Временные файлы индекса '{tmp_index_path}' и '{tmp_id_map_path}' созданы.")
            return tmp_index_path, tmp_id_map_path
        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА [SemanticIndex]: Не удалось записать временные файлы индекса. Ошибка: {e}")
            if os.path.exists(tmp_index_path): os.remove(tmp_index_path)
            if os.path.exists(tmp_id_map_path): os.remove(tmp_id_map_path)
            raise e

    def load_from_disk(self, index_path: str, id_map_path: str) -> bool:
        if os.path.exists(index_path) and os.path.exists(id_map_path):
            print("   [SemanticIndex] -> Загружаю индекс с диска...")
            try:
                self.index = faiss.read_index(index_path)
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
        if claim_id in self.id_map:
            return
        try:
            vector = self.embedding_client.embed_document(claim_text)
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
        if self.index.ntotal == 0:
            return []
        try:
            query_vector = self.embedding_client.embed_query(query_text)
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
        print("   [SemanticIndex] !!! ВНИМАНИЕ: Запущена полная перестройка индекса из Базы Знаний...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = []
        for claim_id, claim_data in knowledge_base.items():
            self._add_claim_internal(claim_id, claim_data['statement'])
        print(f"   [SemanticIndex] <- Индекс перестроен. Всего векторов: {self.index.ntotal}.")
        self.save_to_disk(index_path, id_map_path)

    def _add_claim_internal(self, claim_id: str, claim_text: str):
        try:
            vector = self.embedding_client.embed_document(claim_text)
            vector_np = np.array([vector], dtype=np.float32)
            self.index.add(vector_np)
            self.id_map.append(claim_id)
        except Exception as e:
            print(f"!!! ОШИБКА [SemanticIndex]: Пропускаю claim '{claim_id}' при перестройке.")
            print(f"   -> Тип ошибки: {type(e).__name__}, Сообщение: {e}")