import requests
from bs4 import BeautifulSoup
import re
import json
from typing import Dict, List, Union, Any

def biz_model_analyzer(url_or_text: str) -> Dict[str, Any]:
    """
    Анализирует URL веб-страницы или предоставленный текст для извлечения бизнес-моделей и структур ценообразования.

    Инструмент выполняет семантический анализ на основе набора предопределенных ключевых слов и регулярных выражений,
    связанных с распространенными моделями монетизации. Он определяет такие паттерны, как SaaS-подписки,
    бесплатные тарифы, корпоративные планы и т.д., и возвращает их в структурированном виде.

    Args:
        url_or_text (str): URL веб-страницы для анализа (должен начинаться с 'http://' или 'https://')
                           или предварительно очищенный текстовый контент страницы.

    Returns:
        Dict[str, Any]: Словарь, содержащий статус выполнения и извлеченные данные.
                        В случае успеха:
                        {
                            "status": "success",
                            "data": [
                                {
                                    "model_name": "Название модели",
                                    "description": "Краткое описание найденной модели.",
                                    "evidence": ["Фрагмент текста, подтверждающий находку"]
                                },
                                ...
                            ]
                        }
                        Если модели не найдены, "data" будет пустым списком.

    Raises:
        Exception: Вызывается в случае сетевых ошибок (например, недоступность URL, ошибки HTTP),
                   ошибок парсинга или других непредвиденных проблем в процессе выполнения.
    """
    # --- Начало блока: Определение паттернов для поиска ---
    # Этот блок определяет эвристики для поиска. Он отделен от основной логики для легкого расширения.
    BUSINESS_MODEL_PATTERNS = {
        'saas_subscription': {
            'name': 'SaaS-подписка (Subscription)',
            'description': 'Компания предлагает продукт или услугу по модели подписки, обычно с ежемесячной или ежегодной оплатой.',
            'keywords': ['subscription', 'подписка', 'monthly plan', 'annual plan', 'billed monthly', 'billed annually', 'ежемесячно', 'ежегодно'],
            'regex': [
                r'\$\s?\d+(\.\d{2})?\s?/(month|year|mo|yr)',
                r'\d+\s?₽\s?/(месяц|год|мес)',
                r'per user per month',
                r'за пользователя в месяц'
            ]
        },
        'freemium': {
            'name': 'Бесплатный тариф (Freemium / Free Tier)',
            'description': 'Предлагается базовый уровень продукта или услуги бесплатно, с возможностью платного обновления до расширенной версии.',
            'keywords': ['free plan', 'free tier', 'freemium', 'бесплатный тариф', 'free forever', 'навсегда бесплатно', 'get started for free', 'начните бесплатно'],
            'regex': []
        },
        'enterprise_custom': {
            'name': 'Корпоративный / Индивидуальный расчет (Enterprise / Custom Pricing)',
            'description': 'Для крупных клиентов или особых случаев предлагаются индивидуальные условия и цены, которые обсуждаются с отделом продаж.',
            'keywords': ['enterprise', 'корпоративный', 'contact sales', 'свяжитесь с нами', 'request a quote', 'запросить расчет', 'custom pricing', 'индивидуальный расчет', 'talk to us'],
            'regex': []
        },
        'one_time_purchase': {
            'name': 'Единоразовая покупка (One-Time Purchase)',
            'description': 'Продукт или лицензия на него приобретается один раз за фиксированную плату.',
            'keywords': ['one-time purchase', 'lifetime deal', 'buy now', 'own it forever', 'единоразовый платеж', 'купить навсегда', 'пожизненная лицензия'],
            'regex': [r'lifetime access for \$?\d+']
        },
        'usage_based': {
            'name': 'Оплата по факту использования (Usage-Based / Pay-As-You-Go)',
            'description': 'Стоимость зависит от объема потребленных ресурсов (например, количество API-запросов, использованное хранилище, время работы).',
            'keywords': ['pay-as-you-go', 'usage-based', 'metered billing', 'оплата за использование', 'pay per use'],
            'regex': [
                r'per api call', r'за (1000|тысячу) запросов',
                r'\$\s?\d+(\.\d+)?\s?/gb', r'₽\s?/\s?гб'
            ]
        },
        'marketplace_commission': {
            'name': 'Маркетплейс / Комиссия (Marketplace / Commission)',
            'description': 'Бизнес-модель, основанная на получении комиссии с транзакций между двумя сторонами на платформе.',
            'keywords': ['commission', 'комиссия', 'transaction fee', 'marketplace fee', 'revenue share', 'комиссия с продаж'],
            'regex': [r'\d+(\.\d+)?\s?% commission', r'комиссия\s?\d+(\.\d+)?\s?%']
        }
    }
    # --- Конец блока: Определение паттернов ---

    try:
        text_content = ""
        # Шаг 1: Получение текстового контента
        if url_or_text.strip().startswith(('http://', 'https://')):
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url_or_text.strip(), headers=headers, timeout=20)
            response.raise_for_status()  # Вызовет исключение для кодов 4xx/5xx
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Удаляем скрипты и стили, чтобы они не мешали анализу
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
                
            text_content = soup.get_text(separator=' ', strip=True)
        else:
            text_content = url_or_text

        if not text_content:
            raise Exception("Не удалось получить текстовый контент для анализа.")

        # Шаг 2: Анализ текста на основе паттернов
        lower_text = text_content.lower()
        found_models = []
        detected_model_ids = set()

        # Создаем "окна" текста для поиска контекста
        text_snippets = re.split(r'\n|\.|\!', text_content)
        text_snippets = [s.strip() for s in text_snippets if len(s.strip()) > 10]

        for model_id, pattern_data in BUSINESS_MODEL_PATTERNS.items():
            if model_id in detected_model_ids:
                continue

            found_evidence = []

            # Поиск по ключевым словам
            for keyword in pattern_data['keywords']:
                if f' {keyword.lower()} ' in lower_text:
                    # Поиск контекста для ключевого слова
                    for snippet in text_snippets:
                        if keyword.lower() in snippet.lower():
                            found_evidence.append(snippet)
                            break # Достаточно одного примера для этого ключевого слова
                    if found_evidence:
                        break # Переходим к следующей модели, если нашли подтверждение

            # Поиск по регулярным выражениям, если ключевые слова не найдены
            if not found_evidence:
                for regex_pattern in pattern_data['regex']:
                    matches = re.finditer(regex_pattern, lower_text, re.IGNORECASE)
                    for match in matches:
                        # Найти фрагмент текста вокруг совпадения для контекста
                        start, end = match.span()
                        context_start = max(0, start - 80)
                        context_end = min(len(text_content), end + 80)
                        context_snippet = text_content[context_start:context_end].replace('\n', ' ').strip()
                        found_evidence.append(f"...{context_snippet}...")
                        break
                    if found_evidence:
                        break

            # Если найдено подтверждение, добавляем модель в результат
            if found_evidence:
                detected_model_ids.add(model_id)
                found_models.append({
                    "model_name": pattern_data['name'],
                    "description": pattern_data['description'],
                    "evidence": list(set(found_evidence)) # Убираем дубликаты подтверждений
                })

        # Шаг 3: Формирование и возврат результата
        return {
            "status": "success",
            "data": found_models
        }

    except requests.exceptions.RequestException as e:
        raise Exception(f"Сетевая ошибка при доступе к URL: {e}")
    except Exception as e:
        # Перехватываем все остальные исключения и оборачиваем их
        raise Exception(f"Произошла непредвиденная ошибка при анализе: {e}")