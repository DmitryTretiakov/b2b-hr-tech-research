# core/semantic_index.py
import faiss
import numpy as np
import json
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class SemanticIndex:
    def __init__(self, embedding_model: GoogleGenerativeAIEmbeddings, dimension: int = 768, save_every_n: int = 10):
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.id_map = []
        self.save_every_n = save_every_n
        self._add_counter = 0
        print(f"-> SemanticIndex (FAISS) инициализирован с размерностью {dimension}.")

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
                with open(id_map_path, 'r', encoding='utf-8') as f:
                    self.id_map = json.load(f)
                print(f"   [SemanticIndex] <- Индекс успешно загружен ({self.index.ntotal} векторов).")
                return True
            except Exception as e:
                print(f"!!! ВНИМАНИЕ [SemanticIndex]: Не удалось загрузить индекс. Будет создан новый. Ошибка: {e}")
                return False
        return False

    def add_claim(self, claim_id: str, claim_text: str, index_path: str, id_map_path: str):
        """Добавляет утверждение в индекс и инкрементально сохраняет его."""
        if claim_id in self.id_map:
            return
            
        try:
            vector = self.embedding_model.embed_query(claim_text)
            vector_np = np.array([vector], dtype=np.float32)
            self.index.add(vector_np)
            self.id_map.append(claim_id)
            self._add_counter += 1
            
            if self._add_counter % self.save_every_n == 0:
                self.save_to_disk(index_path, id_map_path)
        except Exception as e:
            print(f"!!! ОШИБКА [SemanticIndex]: Не удалось добавить claim '{claim_id}'. Ошибка: {e}")

    def find_similar_claim_ids(self, query_text: str, top_k: int = 5) -> list[str]:
        """Находит top_k самых похожих claim_id для данного текста."""
        if self.index.ntotal == 0:
            return []

        try:
            query_vector = self.embedding_model.embed_query(query_text)
            query_vector_np = np.array([query_vector], dtype=np.float32)
            distances, indices = self.index.search(query_vector_np, k=min(top_k, self.index.ntotal))
            similar_ids = [self.id_map[i] for i in indices[0]]
            return similar_ids
        except Exception as e:
            print(f"!!! ОШИБКА [SemanticIndex]: Не удалось выполнить поиск. Ошибка: {e}")
            return []

    def rebuild_from_kb(self, knowledge_base: dict, index_path: str, id_map_path: str):
        """АВАРИЙНЫЙ МЕТОД: Полностью перестраивает индекс и сохраняет его."""
        print("   [SemanticIndex] !!! ВНИМАНИЕ: Запущена полная перестройка индекса из Базы Знаний...")
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_map = []
        
        for claim_id, claim_data in knowledge_base.items():
            # Вызываем внутренний метод без логики сохранения, чтобы не сохранять на каждой итерации
            self._add_claim_internal(claim_id, claim_data['statement'])
        
        print(f"   [SemanticIndex] <- Индекс перестроен. Всего векторов: {self.index.ntotal}.")
        self.save_to_disk(index_path, id_map_path)

    def _add_claim_internal(self, claim_id: str, claim_text: str):
        """Внутренний метод для добавления без сохранения."""
        try:
            vector = self.embedding_model.embed_query(claim_text)
            vector_np = np.array([vector], dtype=np.float32)
            self.index.add(vector_np)
            self.id_map.append(claim_id)
        except Exception as e:
            print(f"!!! ОШИБКА [SemanticIndex]: Не удалось добавить claim '{claim_id}' при перестройке. Ошибка: {e}")