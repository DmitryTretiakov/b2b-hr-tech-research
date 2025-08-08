# core/embedding_client.py
import os
import google.generativeai as genai
from core.budget_manager import APIBudgetManager
from google.api_core.exceptions import ResourceExhausted

class GeminiEmbeddingClient:
    def __init__(self, budget_manager: APIBudgetManager):
        self.model_name = "models/gemini-embedding-001"
        self.budget_manager = budget_manager
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY не найден в окружении.")
        genai.configure(api_key=api_key)
        print(f"-> GeminiEmbeddingClient инициализирован для модели {self.model_name}.")

    def get_embedding_dimension(self) -> int:
        """Определяет размерность, делая один контролируемый вызов."""
        if not self.budget_manager.can_i_spend(self.model_name):
            raise ResourceExhausted("Лимит API для эмбеддингов исчерпан.")
        
        result = genai.embed_content(
            model=self.model_name,
            content="test",
            task_type="RETRIEVAL_DOCUMENT",
            output_dimensionality=768
        )
        self.budget_manager.record_spend(self.model_name)
        return len(result['embedding'])

    def embed_query(self, text: str) -> list[float]:
        """Создает эмбеддинг для одного текста с контролируемой размерностью."""
        if not self.budget_manager.can_i_spend(self.model_name):
            raise ResourceExhausted("Лимит API для эмбеддингов исчерпан.")
            
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="RETRIEVAL_QUERY", # Используем правильный тип для поисковых запросов
            output_dimensionality=768
        )
        self.budget_manager.record_spend(self.model_name)
        return result['embedding']

    def embed_document(self, text: str) -> list[float]:
        """Создает эмбеддинг для одного документа с контролируемой размерностью."""
        if not self.budget_manager.can_i_spend(self.model_name):
            raise ResourceExhausted("Лимит API для эмбеддингов исчерпан.")

        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="RETRIEVAL_DOCUMENT", # Используем правильный тип для документов
            output_dimensionality=768
        )
        self.budget_manager.record_spend(self.model_name)
        return result['embedding']