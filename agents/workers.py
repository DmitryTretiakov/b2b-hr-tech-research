# agents/workers.py
import json
from datetime import datetime, timezone
from typing import Dict
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import (
    FactExtractionReport, BatchQualityAssessmentReport, AnalystReport, 
    FinalReport, FinalAnalysisReport, SanityCheckReport,
    FinancialModelArtifact, UserStoryArtifact,
    ReportOutline, ReportSection # <-- ДОБАВИТЬ
)

class BaseResearchAgent(BaseAgent):
    """
    Общий базовый класс для Researcher и Contrarian, работающий по циклу ReAct.
    """
    role_prompt: str = "Твоя роль: Ассистент-исследователь."

    def execute(self, task: dict, model_name: str, user_config: Dict) -> list:
        """
        Выполняет задачу, используя цикл ReAct и зная общую цель проекта.
        """
        print(f"   [{self.__class__.__name__}] -> Задача '{task['task_id']}' (ReAct) на модели {model_name}...")
        
        if not self.tool_registry:
            raise ValueError("ToolRegistry не был предоставлен этому агенту.")

        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        # ИСПРАВЛЕНИЕ: visited_urls берутся из корня состояния, а не из user_config
        visited_urls_str = json.dumps(user_config.get('visited_urls', []), indent=2)
        high_level_context = f"**КОНТЕКСТ ВСЕГО ПРОЕКТА:**\nТы работаешь над достижением следующей главной цели: '{main_goal}'. Твоя текущая задача - это один из шагов на пути к этой цели. Выполняй ее, держа в уме конечный результат.\n\n**УЖЕ ПОСЕЩЕННЫЕ URL (не используй их повторно):**\n{visited_urls_str}"

        available_tools = self.tool_registry.get_tools_for_prompt()
        initial_prompt = f"""{self.role_prompt}
{high_level_context}

**ТВОЯ ТЕКУЩЯЯ ЗАДАЧА:** '{task['description']}'

**ПРОЦЕСС РАБОТЫ (ReAct):**
Ты работаешь в цикле "Мысль -> Действие -> Наблюдение".
1.  **Мысль (Thought):** Проанализируй задачу и реши, какой инструмент использовать.
2.  **Действие (Action):** Верни JSON-объект с вызовом инструмента или с решением завершить работу.

**СПИСОК ИНСТРУМЕНТОВ:**
{available_tools}

**ФОРМАТ ВЫВОДА ДЛЯ ДЕЙСТВИЯ:**
Верни JSON с ключом "tool_to_use" (для вызова инструмента) или "finish" (для завершения).
Пример: `{{ "tool_to_use": {{ "tool_name": "web_search", "args": {{ "query": "..." }} }} }}`

Начинай.
"""
        conversation_history = [initial_prompt]
        max_turns = 5
        
        for i in range(max_turns):
            print(f"      [ReAct] Итерация {i+1}/{max_turns}...")
            full_prompt = "\n".join(conversation_history)
            response = self.llm_client.invoke(model_name, full_prompt)
            conversation_history.append(response.content)
            
            try:
                action_json_str = response.content[response.content.find('{'):response.content.rfind('}')+1]
                action_data = json.loads(action_json_str)

                if "tool_to_use" in action_data:
                    tool_call = action_data["tool_to_use"]
                    tool_name = tool_call.get("tool_name")
                    tool_args = tool_call.get("args", {})
                    
                    # ИСПРАВЛЕНИЕ: Передаем полный объект состояния (который пришел в user_config) в use_tool
                    tool_result = self.tool_registry.use_tool(tool_name, tool_args, user_config)
                    observation = f"OBSERVATION:\n```\n{str(tool_result)[:3000]}\n```"
                    conversation_history.append(observation)
                    print(f"      [ReAct] Инструмент '{tool_name}' выполнен.")
                elif "finish" in action_data:
                    print("      [ReAct] Агент решил завершить сбор информации.")
                    break
                else: raise ValueError("Неверный формат JSON-действия.")
            except Exception as e:
                conversation_history.append(f"OBSERVATION: Ошибка обработки ответа: {e}. Пожалуйста, верни JSON с ключом 'tool_to_use' или 'finish'.")

        final_synthesis_prompt = f"{self.role_prompt}\nПроанализируй всю переписку и извлеки 3-5 ключевых фактов. Заполни все поля.\n\n**ИСТОРИЯ РАБОТЫ:**\n{''.join(conversation_history)}"
        synthesis_model = "gemini-2.5-flash"
        report = invoke_llm_for_json_with_retry(self.llm_client, synthesis_model, "gemini-2.5-flash-lite", final_synthesis_prompt, FactExtractionReport, self.budget_manager)

        if not report or 'extracted_facts' not in report: return []
        for fact in report['extracted_facts']: fact['created_at'] = datetime.now(timezone.utc).isoformat()
        return report['extracted_facts']

class ResearcherAgent(BaseResearchAgent):
    role_prompt: str = "Твоя роль: Ассистент-исследователь. Твоя цель — найти подтверждающие, основные факты по задаче."

class ContrarianAgent(BaseResearchAgent):
    role_prompt: str = "Твоя роль: 'Адвокат Дьявола'. Твоя цель — найти опровержения, критику и провальные кейсы по задаче."

class QualityAssessorAgent(BaseAgent):
    def execute(self, facts_to_assess: list, model_name: str, user_config: Dict) -> dict:
        prompt = f"Твоя роль: Контролер качества. Для КАЖДОГО факта вынеси вердикт по чек-листу (Конкретность, Доказуемость, Полнота) и верни JSON-отчет.\nФАКТЫ:\n{json.dumps(facts_to_assess, ensure_ascii=False, indent=2)}"
        return invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash-lite", prompt, BatchQualityAssessmentReport, self.budget_manager)

class FixerAgent(BaseAgent):
    def execute(self, facts_to_fix: list, model_name: str, user_config: Dict) -> list:
        prompt = f"Твоя роль: Редактор. Исправь факты на основе `feedback`, сохранив `claim_id`. Если исправить невозможно, не включай факт в ответ.\nФАКТЫ:\n{json.dumps(facts_to_fix, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, FactExtractionReport, self.budget_manager)
        return report.get('extracted_facts', [])

class SanityCheckCritic(BaseAgent):
    def execute(self, facts_to_check: list, model_name: str, user_config: Dict) -> list:
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        prompt = f"**Главная цель проекта:** {main_goal}\n\nТвоя роль: Старший аналитик. Проверь факты на коммерческую релевантность для достижения главной цели и отсутствие 'воды'. Верни JSON со списком `verified_claim_ids` тех, кто прошел проверку.\nФАКТЫ:\n{json.dumps(facts_to_check, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, SanityCheckReport, self.budget_manager)
        verified_ids = set(report.get('verified_claim_ids', []))
        return [fact for fact in facts_to_check if fact['claim_id'] in verified_ids]

class AnalystAgent(BaseAgent):
    def execute_reflection(self, knowledge_base: dict, model_name: str, user_config: Dict) -> dict:
        prompt = f"Твоя роль: Старший аналитик. Проанализируй Базу Знаний и предоставь краткую сводку для планировщика: 3-5 ключевых инсайтов и 2-3 пробела в данных.\nБАЗА ЗНАНИЙ:\n{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, AnalystReport, self.budget_manager)
        return {"status": "SUCCESS", "data": report}

    def execute_final_synthesis(self, knowledge_base: dict, model_name: str, user_config: Dict) -> dict:
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        prompt = f"**КОНТЕКСТ ПРОЕКТА:** {main_goal}\n\n**ТВОЯ РОЛЬ:** Ведущий аналитик-стратег.\n**ТВОЯ ЗАДАЧА:** Превратить базу фактов в структурированный аналитический документ, который поможет достичь цели проекта. Заполни ВСЕ поля JSON-схемы.\n\n**БАЗА ЗНАНИЙ:**\n{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}"
        return invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, FinalAnalysisReport, self.budget_manager)

# --- НОВЫЕ АГЕНТЫ-СПЕЦИАЛИСТЫ ---

class FinancialModelAgent(BaseAgent):
    """Генерирует артефакт с базовой финансовой моделью."""
    def execute(self, task: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [FinancialModelAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        knowledge_base = task.get("knowledge_base", {}) # Предполагаем, что KB передается в задаче
        
        prompt = f"""
**КОНТЕКСТ ПРОЕКТА:** {main_goal}

**ТВОЯ РОЛЬ:** Финансовый аналитик, специализирующийся на HR-Tech SaaS продуктах.
**ТВОЯ ЗАДАЧА:** {task['description']}

**ИНСТРУКЦИИ:**
1.  Внимательно изучи всю доступную Базу Знаний. Найди в ней факты, касающиеся:
    - Средних зарплат IT-специалистов (для расчета CAPEX/OPEX).
    - Стоимости аналогичных продуктов на рынке (для прогноза Revenue).
    - Типовых бизнес-моделей (подписка, per-seat и т.д.).
2.  Сформулируй ключевые допущения (`key_assumptions`), на которых будет строиться твоя модель. Будь реалистичен.
3.  Создай Markdown-таблицу (`calculations_table_markdown`) с базовым расчетом юнит-экономики и прогнозом на 3 года.
4.  Напиши краткий, но емкий вывод (`summary_conclusion`) по результатам.

**БАЗА ЗНАНИЙ ДЛЯ АНАЛИЗА:**
```json
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
```
"""
        artifact = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, FinancialModelArtifact, self.budget_manager)
        return artifact

class ProductManagerAgent(BaseAgent):
    """Генерирует артефакт с User Stories для дорожной карты продукта."""
    def execute(self, task: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [ProductManagerAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
        product_vision = user_config.get("project_context", {}).get("product_vision", "Видение продукта не определено.")
        knowledge_base = task.get("knowledge_base", {})

        prompt = f"""
**ВИДЕНИЕ ПРОДУКТА:** {product_vision}

**ТВОЯ РОЛЬ:** Опытный AI Product Manager.
**ТВОЯ ЗАДАЧА:** {task['description']}

**ИНСТРУКЦИИ:**
1.  Основываясь на видении продукта и данных из Базы Знаний, определи ключевые фичи для указанного эпика.
2.  Для каждой фичи напиши User Story в классическом формате: "Как <роль>, я хочу <действие>, чтобы <ценность>".
3.  Сгруппируй их в единый JSON-артефакт.

**БАЗА ЗНАНИЙ ДЛЯ КОНТЕКСТА:**
```json
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
```
"""
        artifact = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, UserStoryArtifact, self.budget_manager)
        return artifact

class ReportWriterAgent(BaseAgent):
    """Собирает финальный отчет из написанных секций, добавляя введение и заключение."""
    def execute(self, task: dict, model_name: str, user_config: Dict) -> str:
        """Принимает все части отчета и компилирует их в финальный документ."""
        print(f"   [ReportWriterAgent] -> Компилирую финальный отчет на модели {model_name}...")
        profile = user_config.get("user_context", {}).get("profile", "Профиль не определен.")
        drafted_sections = task.get("drafted_sections", [])
        report_title = task.get("report_title", "Аналитический отчет")

        # Собираем все секции в один большой текст
        full_draft = f"# {report_title}\n\n"
        for i, section in enumerate(drafted_sections):
            full_draft += f"## {i+1}. {section.get('section_title', '')}\n\n{section.get('markdown_content', '')}\n\n"

        prompt = f"""
**КОНТЕКСТ АУДИТОРИИ:** {profile}

**ТВОЯ РОЛЬ:** Главный редактор.
**ТВОЯ ЗАДАЧА:** Взять черновик отчета, состоящий из готовых секций, и довести его до совершенства.

**ЧЕРНОВИК ОТЧЕТА:**
---
{full_draft}
---

**ИНСТРУКЦИИ:**
1.  Напиши краткое, но емкое **Введение (Executive Summary)**, которое обобщает ключевые выводы всего документа.
2.  Проверь стилистическую целостность текста.
3.  Напиши сильное **Заключение**, которое подводит итоги и предлагает следующие шаги.
4.  Собери все вместе (Введение + Текст секций + Заключение) в один финальный Markdown-документ.

Верни только финальный `markdown_content`.
"""
        report = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, FinalReport, self.budget_manager)
        return report.get('markdown_content', '')


class ReviserAgent(BaseAgent):
    """
    Агент-критик, который оценивает полноту и релевантность собранной информации
    в середине исследовательской фазы и корректирует курс.
    """
    def execute(self, task: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [ReviserAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
        
        # Извлекаем все необходимые данные из задачи, которую для нас сформировал orchestrator
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        knowledge_base = task.get("knowledge_base", {})
        remaining_tasks = task.get("remaining_tasks", [])

        prompt = f"""
**ТВОЯ РОЛЬ:** Ты - Ведущий Исследователь-Стратег и внутренний критик. Твоя задача - не выполнять поиск, а анализировать уже проделанную работу и корректировать дальнейший курс.

**ГЛАВНАЯ ЦЕЛЬ ПРОЕКТА:**
{main_goal}

**УЖЕ СОБРАННЫЕ ФАКТЫ (ТЕКУЩАЯ БАЗА ЗНАНИЙ):**
```json
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
```

**ЗАДАЧI, КОТОРЫЕ ЕЩЕ ОСТАЛОСЬ ВЫПОЛНИТЬ В ЭТОЙ ФАЗЕ:**
{json.dumps(remaining_tasks, ensure_ascii=False, indent=2)}

**ТВОЯ ЗАДАЧА - ПРОВЕСТИ РЕВИЗИЮ:**
1.  **Оцени Достаточность:** Достаточно ли уже собранных фактов для ответа на главный вопрос проекта? Не ушли ли мы в сторону?
2.  **Найди "Слепые Зоны":** Каких критически важных данных все еще не хватает? Есть ли в собранной информации предвзятость (например, только положительные отзывы)?
3.  **Прими Решение:**
    - Если информация полна и релевантна, установи `is_sufficient: true`.
    - Если нужны доработки, установи `is_sufficient: false`, дай четкий `feedback` и предложи конкретные формулировки для новых задач в `new_task_suggestions`.

Верни результат в виде JSON, соответствующего схеме `RevisionReport`.
"""
        report = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, RevisionReport, self.budget_manager)
        return report
    
class OutlineAgent(BaseAgent):
    """Генерирует детальный план (оглавление) для финального отчета."""
    def execute(self, task: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [OutlineAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        knowledge_base = task.get("knowledge_base", {})

        prompt = f"""
**ГЛАВНАЯ ЦЕЛЬ ПРОЕКТА:** {main_goal}

**ТВОЯ РОЛЬ:** Ведущий аналитик и редактор.
**ТВОЯ ЗАДАЧА:** Создать детальный, логически выстроенный план (оглавление) для финального аналитического отчета.

**ИНСТРУКЦИИ:**
1.  Изучи главную цель и всю Базу Знаний.
2.  Предложи броский, но профессиональный заголовок (`title`) для всего отчета.
3.  Разбей отчет на логические секции (`sections`). Для каждой секции укажи `section_title` и `section_description` (краткое описание того, какие ключевые выводы и факты должны быть в этой секции). План должен включать введение, основную часть с анализом и заключение.

**БАЗА ЗНАНИЙ ДЛЯ АНАЛИЗА:**
```json
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
```
"""
        outline = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash", prompt, ReportOutline, self.budget_manager)
        return outline

class SectionWriterAgent(BaseAgent):
    """Пишет текст для одной конкретной секции отчета по заданному плану."""
    def execute(self, task: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [SectionWriterAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
        section_to_draft = task.get("section_to_draft", {})
        knowledge_base = task.get("knowledge_base", {})

        prompt = f"""
**ТВОЯ РОЛЬ:** Эксперт-аналитик и копирайтер.
**ТВОЯ ЗАДАЧА:** Написать текст для ОДНОЙ секции отчета, строго следуя плану.

**ПЛАН СЕКЦИИ:**
- **Название:** {section_to_draft.get('section_title')}
- **Что раскрыть:** {section_to_draft.get('section_description')}

**ИНСТРУКЦИИ:**
1.  Используй информацию из Базы Знаний для написания текста.
2.  После каждого утверждения, подкрепленного фактом из Базы Знаний, вставь маркер цитирования `[CITE:claim_id]`.
3.  Не выходи за рамки плана для данной секции.
4.  Верни только готовый текст в формате Markdown.

**БАЗА ЗНАНИЙ ДЛЯ ИСПОЛЬЗОВАНИЯ:**
```json
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
```
"""
        section = invoke_llm_for_json_with_retry(self.llm_client, model_name, "gemini-2.5-flash-lite", prompt, ReportSection, self.budget_manager)
        return section

