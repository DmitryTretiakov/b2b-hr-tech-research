# agents/expert_team.py
import json
from utils.helpers import parse_json_from_response, format_search_results_for_llm

class ExpertTeam:
    """
    Управляет командой экспертов. Получает задачу и ОБЩИЙ КОНТЕКСТ,
    проводит исследование, аудит и возвращает список верифицированных "Утверждений".
    """
    def __init__(self, llms: dict, search_agent):
        self.llms = llms
        self.search_agent = search_agent
        print("-> Команда Экспертов сформирована.")

    def execute_task(self, task: dict, world_model_context: dict) -> list:
        """Основной метод, запускающий полный цикл работы над одной задачей."""
        assignee = task['assignee']
        description = task['description']
        goal = task['goal']
        print(f"\n--- Эксперт {assignee}: Приступаю к задаче '{description}' ---")

        try:
            # 1. Декомпозиция задачи
            search_queries = self._decompose_task(assignee, description, goal, world_model_context)
            if not search_queries: return []

            # 2. Поиск и форматирование результатов
            raw_results = [self.search_agent.search(q) for q in search_queries]
            search_results_str = "\n".join([format_search_results_for_llm(r) for r in raw_results])
            
            if not search_results_str.strip() or "Поиск не дал результатов" in search_results_str:
                print(f"!!! Эксперт {assignee}: Поиск не дал результатов. Задача не может быть выполнена.")
                return []

            # 3. Написание черновика "Утверждений"
            draft_claims = self._create_draft_claims(assignee, description, goal, search_results_str, world_model_context)
            if not draft_claims: return []

            # 4. Аудит
            vulnerabilities = self._audit_claims(draft_claims, world_model_context)
            
            # 5. Финализация
            final_claims = self._finalize_claims(assignee, description, search_results_str, draft_claims, vulnerabilities, world_model_context)
            if not final_claims: return []

            print(f"--- Эксперт {assignee}: Задача выполнена, сгенерировано {len(final_claims)} верифицированных утверждений. ---")
            return final_claims

        except Exception as e:
            print(f"!!! КРИТИЧЕСКАЯ ОШИБКА в ExpertTeam при выполнении задачи '{description}': {e}")
            return []

    def _get_llm_for_expert(self, assignee: str):
        """Выбирает модель в зависимости от роли эксперта."""
        if assignee in ["HR_Expert", "ProductOwnerAgent"]:
            return self.llms["expert_flash"]
        else:
            return self.llms["expert_lite"]

    def _decompose_task(self, assignee: str, description: str, goal: str, context: dict) -> list:
        """Шаг 1: Генерирует поисковые запросы с учетом общей цели."""
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}

**КОНТЕКСТ:**
Ты - ассистент-исследователь для эксперта '{assignee}'.
Цель твоего эксперта: {goal}
Текущая задача эксперта: {description}

**ТВОЯ ЗАДАЧА:**
Сгенерируй от 4 до 6 максимально конкретных и разнообразных поисковых запросов на русском языке, которые помогут эксперту найти ДОКАЗАТЕЛЬСТВА и ФАКТЫ для выполнения его задачи. Думай о том, какие запросы помогут найти цифры, отчеты и мнения экспертов.

**ПРИМЕР ВЫВОДА:**
```json
[
  "отзывы сотрудников о LMS Websoft",
  "стоимость внедрения Websoft HCM для 500 человек",
  "сравнение Websoft HCM и 1С ЗУП КОРП"
]
```

Твой результат:"""
        llm = self._get_llm_for_expert(assignee)
        response = llm.invoke(prompt)
        queries = parse_json_from_response(response.content)
        
        if queries and isinstance(queries, list):
            print(f"   [Эксперт {assignee}] Шаг 1/5: Сгенерированы поисковые запросы: {queries}")
            return queries
        else:
            print(f"!!! Эксперт {assignee}: Не удалось сгенерировать поисковые запросы.")
            return []

    def _create_draft_claims(self, assignee: str, description: str, goal: str, search_results: str, context: dict) -> list:
        """Шаг 2: Создает черновой список "Утверждений" с учетом общей цели."""
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}

**ТВОЯ РОЛЬ И ЗАДАЧА:**
Ты - {assignee}. Твоя цель - {goal}.
Твоя текущая задача - проанализировать результаты поиска по теме '{description}' и сформулировать несколько ключевых "Утверждений" (Claims).
Каждое утверждение должно быть максимально конкретным, основанным на данных и РЕЛЕВАНТНЫМ для ОБЩЕЙ МИССИИ ПРОЕКТА.

**ПРАВИЛА ФОРМАТИРОВАНИЯ:**
Ты ОБЯЗАН вернуть результат в виде списка JSON-объектов.
- `claim_id`: Создай короткий, уникальный и информативный ID на английском (например, 'websoft_customization_weakness').
- `statement`: Четкий вывод или факт.
- `value`: Конкретное значение (цифра, текст, список).
- `source_link`: Прямая ссылка на самый релевантный источник.
- `source_quote`: Прямая цитата из источника, подтверждающая утверждение.
- `confidence_score`: Твоя оценка от 0.0 до 1.0, насколько ты доверяешь этому источнику.
- `status`: Всегда "UNVERIFIED" на этом шаге.

**ПРИМЕР ВЫВОДА:**
```json
[
  {{
    "claim_id": "ispring_learn_price_50_users",
    "statement": "Стоимость годовой подписки на iSpring Learn для 50 пользователей.",
    "value": "137 700 рублей",
    "source_link": "https://www.ispring.ru/ispring-learn/pricing",
    "source_quote": "iSpring Learn 50 пользователей, 1 год, 137 700 ₽",
    "confidence_score": 1.0,
    "status": "UNVERIFIED"
  }}
]
```

**РЕЗУЛЬТАТЫ ПОИСКА ДЛЯ АНАЛИЗА:**
---
{search_results}
---

Твой результат:"""
        llm = self._get_llm_for_expert(assignee)
        response = llm.invoke(prompt)
        claims = parse_json_from_response(response.content)

        if claims and isinstance(claims, list):
            print(f"   [Эксперт {assignee}] Шаг 2/5: Создан черновик из {len(claims)} утверждений.")
            return claims
        else:
            print(f"!!! Эксперт {assignee}: Не удалось создать черновик утверждений.")
            return []

    def _audit_claims(self, claims: list, context: dict) -> dict:
        """Шаг 3: "Враждебный Аудитор" проверяет утверждения с учетом общей цели."""
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}

**ТВОЯ РОЛЬ И ЗАДАЧА:**
Твоя Роль: "Враждебный Аудитор". Ты не доверяешь ничему.
Твоя Задача: Тебе предоставлен список "Утверждений", сделанных другим AI. Твоя цель - найти в них логические ошибки, слабые места или недостаток доказательств, особенно те, которые могут ввести в заблуждение при достижении ОБЩЕЙ МИССИИ ПРОЕКТА.

Для каждого `claim_id` верни список текстовых "уязвимостей". Если уязвимостей нет, верни пустой список.

**ПРИМЕР ВЫВОДА:**
```json
{{
  "ispring_learn_price_50_users": [
    "Источник является официальным сайтом, цена может быть маркетинговой и не включать стоимость внедрения. Это критично для нашей бизнес-модели.",
    "Данные актуальны, но не указана дата проверки цены. Для финансового прогноза это недопустимо."
  ],
  "another_claim_id": []
}}
```

**УТВЕРЖДЕНИЯ ДЛЯ АУДИТА:**
---
{json.dumps(claims, ensure_ascii=False, indent=2)}
---

Твой аудиторский отчет:"""
        response = self.llms["expert_flash"].invoke(prompt) # Аудитор всегда "умный"
        vulnerabilities = parse_json_from_response(response.content)

        if vulnerabilities and isinstance(vulnerabilities, dict):
            print(f"   [Аудитор] Шаг 3/5: Проведена проверка, найдены уязвимости.")
            return vulnerabilities
        else:
            print(f"!!! Аудитор: Не удалось провести аудит.")
            return {}

    def _finalize_claims(self, assignee: str, description: str, search_results: str, draft_claims: list, vulnerabilities: dict, context: dict) -> list:
        """Шаг 4: Эксперт дорабатывает утверждения с учетом критики и общей цели."""
        prompt = f"""**ОБЩАЯ МИССИЯ ПРОЕКТА:**
{context['static_context']['main_goal']}

**ТВОЯ РОЛЬ И ЗАДАЧА:**
Ты - {assignee}.
Твоя Задача: Пересмотреть свой черновик "Утверждений" по теме '{description}' с учетом отчета от "Враждебного Аудитора".
Исправь свои утверждения, уточни формулировки и, что **ОЧЕНЬ ВАЖНО**, скорректируй `confidence_score` на основе критики (например, если источник рекламный, понизь уверенность). Твои финальные утверждения должны быть максимально точными и полезными для ОБЩЕЙ МИССИИ ПРОЕКТА.

**ОРИГИНАЛЬНЫЕ РЕЗУЛЬТАТЫ ПОИСКА:**
---
{search_results}
---
**ТВОЙ ЧЕРНОВИК УТВЕРЖДЕНИЙ:**
---
{json.dumps(draft_claims, ensure_ascii=False, indent=2)}
---
**ОТЧЕТ АУДИТОРА (УЯЗВИМОСТИ):**
---
{json.dumps(vulnerabilities, ensure_ascii=False, indent=2)}
---

Твой финальный, исправленный результат (в формате списка JSON-объектов):"""
        llm = self._get_llm_for_expert(assignee)
        response = llm.invoke(prompt)
        final_claims = parse_json_from_response(response.content)

        if final_claims and isinstance(final_claims, list):
            print(f"   [Эксперт {assignee}] Шаг 4/5: Утверждения финализированы с учетом аудита.")
            # Шаг 5: Присваиваем статус VERIFIED
            for claim in final_claims:
                claim['status'] = 'VERIFIED'
            print(f"   [Эксперт {assignee}] Шаг 5/5: Статус утверждений обновлен на VERIFIED.")
            return final_claims
        else:
            print(f"!!! Эксперт {assignee}: Не удалось финализировать утверждения.")
            return []
