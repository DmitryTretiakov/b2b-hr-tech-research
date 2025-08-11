# utils/text_processing.py
import tiktoken
from typing import List

# Инициализируем токенизатор один раз для эффективности
try:
    # "cl100k_base" - это стандартный токенизатор, который хорошо работает для моделей GPT-3.5/4 и многих других, включая Gemini.
    tokenizer = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Резервный вариант, если основная модель не найдена
    tokenizer = tiktoken.get_encoding("p50k_base")

def count_tokens(text: str) -> int:
    """
    Надежно подсчитывает количество токенов в строке.
    """
    if not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

def chunk_text_by_tokens(text: str, chunk_size: int) -> List[str]:
    """
    Разделяет один большой текст на чанки заданного размера в токенах.
    """
    if count_tokens(text) <= chunk_size:
        return [text]

    # Простой, но надежный способ чанкизации
    words = text.split(' ')
    chunks = []
    current_chunk_words = []
    current_token_count = 0

    for word in words:
        word_token_count = count_tokens(word + ' ')
        if current_token_count + word_token_count > chunk_size:
            chunks.append(" ".join(current_chunk_words))
            current_chunk_words = [word]
            current_token_count = word_token_count
        else:
            current_chunk_words.append(word)
            current_token_count += word_token_count
    
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
        
    return chunks