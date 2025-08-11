# core/context_compressor.py
import json
from typing import Dict, List
from pydantic import BaseModel, Field

from core.llm_client import LLMClient
from core.budget_manager import APIBudgetManager
from utils.text_processing import count_tokens, chunk_text_by_tokens
from utils.helpers import invoke_llm_for_json_with_retry

class Summary(BaseModel):
    summary_text: str = Field(description="Связная аналитическая сводка, сгенерированная на основе исходных данных.")

class ContextCompressor:
    """
    Реализует адаптивный, многоуровневый алгоритм сжатия контекста.
    """
    def __init__(self, llm_client: LLMClient, budget_manager: APIBudgetManager):
        self.llm_client = llm_client
        self.budget_manager = budget_manager
        self.level_1_model = "gemini-2.5-flash-lite" # Быстрая и дешевая модель для первого прохода
        self.level_2_model = "gemini-2.5-flash"     # Чуть более мощная модель для финальной сборки

    def compress(self, knowledge_base: Dict, task_description: str, target_token_count: int = 15000) -> str:
        """
        Основной метод. Сжимает knowledge_base до целевого размера.
        """
        print("\n--- Узел: Context Compressor ---")
        
        # Шаг 0: Предварительная оценка
        facts_as_text_list = self._format_kb_as_text(knowledge_base)
        full_text = "\n\n---\n\n".join(facts_as_text_list)
        initial_tokens = count_tokens(full_text)
        print(f"   [Compressor] Начальный размер контекста: {initial_tokens} токенов.")

        if initial_tokens <= target_token_count:
            print("   [Compressor] <- Контекст уже в пределах лимита. Сжатие не требуется.")
            return full_text

        # Шаг 1: Группировка по задачам и чанкизация
        task_groups = self._group_facts_by_task(knowledge_base)
        level_1_chunks = []
        for task_id, facts in task_groups.items():
            task_text = "\n\n---\n\n".join(facts)
            # Делим текст задачи на чанки, если он слишком большой
            chunks_for_task = chunk_text_by_tokens(task_text, chunk_size=30000)
            level_1_chunks.extend(chunks_for_task)
        
        print(f"   [Compressor] Контекст разделен на {len(level_1_chunks)} чанков для 1-го уровня сжатия.")

        # Шаг 2: Первый уровень сжатия (Map)
        level_1_summaries = [self._summarize_chunk(chunk, task_description, 2000) for chunk in level_1_chunks]
        
        # Шаг 3: Промежуточная оценка и, при необходимости, второй уровень (Reduce)
        summaries_text = "\n\n---\n\n".join(level_1_summaries)
        level_1_tokens = count_tokens(summaries_text)
        print(f"   [Compressor] Размер контекста после 1-го уровня: {level_1_tokens} токенов.")

        if level_1_tokens <= target_token_count:
            print("   [Compressor] <- Целевой размер достигнут после 1-го уровня.")
            return summaries_text

        # Запускаем 2-й уровень
        print("   [Compressor] Запускаю 2-й уровень сжатия...")
        compression_ratio = level_1_tokens / target_token_count
        level_2_target_size = int(2000 / compression_ratio) # Адаптивно уменьшаем целевой размер сводки
        
        level_2_chunks = chunk_text_by_tokens(summaries_text, chunk_size=30000)
        level_2_summaries = [self._summarize_chunk(chunk, task_description, level_2_target_size, is_final=True) for chunk in level_2_chunks]

        final_context = "\n\n---\n\n".join(level_2_summaries)
        final_tokens = count_tokens(final_context)
        print(f"   [Compressor] <- Финальный размер контекста: {final_tokens} токенов.")
        return final_context

    def _summarize_chunk(self, chunk_text: str, task_description: str, target_size: int, is_final: bool = False) -> str:
        """Вызывает LLM для сжатия одного чанка текста."""
        level_str = "финальную" if is_final else ""
        prompt = f"""
Твоя роль: Аналитик-архивариус. Ты готовишь материалы для старшего аналитика, который работает над следующей задачей:
**ЦЕЛЬЕВАЯ ЗАДАЧА:** "{task_description}"

Исходя из этой цели, проанализируй текст ниже и напиши связную, {level_str} аналитическую сводку объемом примерно {target_size} токенов.
**Критически важно:** в сводке должны быть сохранены все ключевые сущности: цифры, факты, названия, даты, а также **контекст** — откуда была взята информация (ссылки на источники). Не выбрасывай детали, а структурируй и обобщи их.

Твой ответ должен быть только текстом сводки. Не пиши ничего вроде 'Конечно, вот сводка:'. Начинай сразу с первого предложения.

**ИСХОДНЫЕ ДАННЫЕ ДЛЯ АНАЛИЗА:**
---
{chunk_text}
---
"""
        model = self.level_2_model if is_final else self.level_1_model
        result = invoke_llm_for_json_with_retry(
            self.llm_client, model, self.level_1_model, prompt, Summary, self.budget_manager
        )
        return result.get("summary_text", "")

    def _format_kb_as_text(self, knowledge_base: Dict) -> List[str]:
        """Преобразует каждый факт из KB в читаемую строку с метаданными."""
        formatted_facts = []
        for fact_id, fact_data in knowledge_base.items():
            text = (
                f"ID Факта: {fact_id}\n"
                f"Утверждение: {fact_data.get('statement')}\n"
                f"Источник: {fact_data.get('source_link')}\n"
                f"Дата создания: {fact_data.get('created_at')}"
            )
            formatted_facts.append(text)
        return formatted_facts

    def _group_facts_by_task(self, knowledge_base: Dict) -> Dict[str, List[str]]:
        """Группирует факты по ID задачи, из которой они были извлечены."""
        groups = {}
        for fact_id, fact_data in knowledge_base.items():
            # Извлекаем ID задачи из ID факта (например, fact_research_01 -> research_01)
            task_id = "_".join(fact_id.split('_')[1:])
            if task_id not in groups:
                groups[task_id] = []
            
            fact_text = (
                f"ID Факта: {fact_id}\n"
                f"Утверждение: {fact_data.get('statement')}\n"
                f"Источник: {fact_data.get('source_link')}\n"
                f"Дата создания: {fact_data.get('created_at')}"
            )
            groups[task_id].append(fact_text)
        return groups