# agents/workers.py
import json
import sys
import traceback
from datetime import datetime, timezone
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from agents.base_agent import BaseAgent
from core.context_compressor import ContextCompressor
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import (
    FactExtractionReport, BatchQualityAssessmentReport, AnalystReport, 
    FinalReport, FinalAnalysisReport, RevisionReport, SanityCheckReport,
    FinancialModelArtifact, UserStoryArtifact,
    ReportOutline, ReportSection
)

class ToolChoice(BaseModel):
    """Модель для выбора одного инструмента и его аргументов."""
    thought: str = Field(description="Краткое объяснение, почему выбран именно этот инструмент.")
    tool_name: str = Field(description="Название одного инструмента для использования из списка доступных.")
    args: Dict = Field(description="Словарь с аргументами для вызова выбранного инструмента.")

class SingleStepToolAgent(BaseAgent):
    """
    Простой и надежный агент, который выполняет ровно одно действие:
    1. Выбирает лучший инструмент для задачи.
    2. Выполняет его.
    3. Возвращает результат.
    Идеально подходит для работы с менее мощными моделями.
    """
    def execute(self, task: dict, model_name: str, state: dict) -> list:
        try:
            print(f"   [SingleStepToolAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
            if not self.tool_registry:
                raise ValueError("ToolRegistry не был предоставлен этому агенту.")

            available_tools = self.tool_registry.get_tools_for_prompt()
            
            prompt = f"""
**ТВОЯ РОЛЬ:** Ты - эффективный ассистент. Твоя цель - выбрать ОДИН наиболее подходящий инструмент для выполнения задачи.

**ЗАДАЧА:**
{task['description']}

**СПИСОК ДОСТУПНЫХ ИНСТРУМЕНТОВ:**
{available_tools}

**ИНСТРУКЦИИ:**
1. Проанализируй задачу.
2. Выбери из списка ОДИН инструмент, который лучше всего подходит для ее решения.
3. Сформируй необходимые аргументы для этого инструмента.
4. Верни свой выбор в виде JSON-объекта.
"""
            
            # Шаг 1: LLM выбирает инструмент
            tool_choice_dict = invoke_llm_for_json_with_retry(
                self.llm_client, model_name, "gemini-2.5-flash-lite", prompt, ToolChoice, self.budget_manager
            )

            if not tool_choice_dict:
                print("      [SingleStepToolAgent] !!! Не удалось получить выбор инструмента от LLM.")
                return []

            print(f"      [SingleStepToolAgent] LLM выбрал инструмент: '{tool_choice_dict.get('tool_name')}' с мыслью: '{tool_choice_dict.get('thought')}'")

            # Шаг 2: Выполняем выбранный инструмент
            tool_name = tool_choice_dict.get('tool_name')
            tool_args = tool_choice_dict.get('args', {})
            
            result = self.tool_registry.use_tool(tool_name, tool_args, state)
            
            # Шаг 3: Форматируем результат как факт
            fact = {
                "claim_id": f"fact_{task['task_id']}",
                "statement": f"Результат выполнения задачи '{task['description']}': {json.dumps(result, ensure_ascii=False)}",
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
                "source_link": f"tool://{tool_name}",
                "source_quote": json.dumps(tool_args, ensure_ascii=False)
            }
            return [fact]

        except Exception as e:
            print("\n" + "="*80, file=sys.stderr)
            print(f"!!! КРИТИЧЕСКИЙ СБОЙ ВНУТРИ АГЕНТА '{self.__class__.__name__}'", file=sys.stderr)
            print(f"    Задача: {task.get('task_id')}", file=sys.stderr)
            print(f"    Тип ошибки: {type(e).__name__} - {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            sys.stderr.flush()
            return []

class BaseResearchAgent(BaseAgent):
    """
    Общий базовый класс для Researcher и Contrarian, использующий стандартный
    и надежный AgentExecutor для выполнения цикла ReAct.
    """
    role_prompt: str = "Твоя роль: Ассистент-исследователь."

    def execute(self, task: dict, model_name: str, state: dict) -> list:
        """
        Выполняет задачу, используя стандартный AgentExecutor для максимальной надежности.
        """
        try:
            print(f"   [{self.__class__.__name__}] -> Задача '{task['task_id']}' (ReAct) на модели {model_name}...")
            
            if not self.tool_registry:
                raise ValueError("ToolRegistry не был предоставлен этому агенту.")

            # Получаем необходимый контекст из состояния
            main_goal = state.get("user_config", {}).get("user_context", {}).get("main_goal", "Цель не определена.")
            visited_urls_str = json.dumps(state.get('visited_urls', []), indent=2)
            
            # Формируем промпт, совместимый с create_react_agent
            template = f"""
{self.role_prompt}

**КОНТЕКСТ ВСЕГО ПРОЕКТА:**
Ты работаешь над достижением следующей главной цели: '{main_goal}'.

**УЖЕ ПОСЕЩЕННЫЕ URL (не используй их повторно):**
{visited_urls_str}

**ТВОЯ ТЕКУЩАЯ ЗАДАЧА:**
{{input}}

**ПРОЦЕСС РАБОТЫ (ReAct):**
Ты должен отвечать, используя формат JSON. Твой JSON должен содержать либо ключ 'thought' и 'action' для использования инструмента, либо ключ 'thought' и 'final_answer' для завершения работы.

1.  **Thought:** Кратко опиши свой план действий.
2.  **Action:** Выбери один из доступных инструментов.
    - `tool_name`: Название инструмента из списка.
    - `args`: Словарь с аргументами для инструмента.
3.  **Final Answer:** Когда ты нашел ответ на задачу, предоставь его здесь.

**ДОСТУПНЫЕ ИНСТРУМЕНТЫ:**
{{tools}}

{{agent_scratchpad}}
"""
            prompt = PromptTemplate.from_template(template)
            llm = self.llm_client._get_model_instance(model_name)
            tools = list(self.tool_registry.tools.values())
            
            # Создаем и конфигурируем стандартный агент
            agent = create_react_agent(llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=tools, 
                verbose=True, 
                handle_parsing_errors="Пожалуйста, исправь свой предыдущий вывод. Он должен быть валидным JSON с ключами 'action' или 'final_answer'.",
                max_iterations=7
            )

            print("      [ReAct] Запускаю стандартный AgentExecutor...")
            result = agent_executor.invoke({"input": task['description']})

            if not result or 'output' not in result or not result['output']:
                print("      [ReAct] !!! AgentExecutor завершил работу без финального ответа.")
                return []

            final_answer = result['output']
            print(f"      [ReAct] <- AgentExecutor успешно завершен. Финальный ответ: {final_answer[:200]}...")

            # Создаем один факт на основе финального ответа агента
            # В более сложных сценариях здесь может быть вызов LLM для парсинга ответа на несколько фактов
            fact = {
                "claim_id": f"fact_{task['task_id']}",
                "statement": final_answer,
                "version": 1,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status": "ACTIVE",
                "source_link": "Generated by ReAct agent",
                "source_quote": "N/A"
            }
            return [fact]

        except Exception as e:
            print("\n" + "="*80, file=sys.stderr)
            print(f"!!! КРИТИЧЕСКИЙ СБОЙ ВНУТРИ АГЕНТА '{self.__class__.__name__}'", file=sys.stderr)
            print(f"    Задача: {task.get('task_id')}", file=sys.stderr)
            print(f"    Тип ошибки: {type(e).__name__} - {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("="*80 + "\n", file=sys.stderr)
            sys.stderr.flush()
            # Возвращаем пустой список, чтобы orchestrator мог корректно обработать сбой
            return []

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
        report = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, SanityCheckReport, self.budget_manager)        
        verified_ids = set(report.get('verified_claim_ids', []))
        return [fact for fact in facts_to_check if fact['claim_id'] in verified_ids]

class AnalystAgent(BaseAgent):
    # === ИЗМЕНЕНИЕ НАЧАТО: Принимаем и храним компрессор ===
    def __init__(self, llm_client, budget_manager, context_compressor: ContextCompressor):
        super().__init__(llm_client, budget_manager)
        self.context_compressor = context_compressor
    # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    def execute_reflection(self, knowledge_base: dict, model_name: str, user_config: Dict) -> dict:
        # === ИЗМЕНЕНИЕ НАЧАТО: Используем компрессор ===
        task_desc = "Проанализировать Базу Знаний, чтобы найти 3-5 ключевых инсайтов и 2-3 пробела в данных для планирования следующего шага."
        compressed_kb = self.context_compressor.compress(knowledge_base, task_desc)
        
        prompt = f"Твоя роль: Старший аналитик. Проанализируй следующую сводку из Базы Знаний и предоставь краткий отчет для планировщика: 3-5 ключевых инсайтов и 2-3 пробела в данных.\nСВОДКА:\n{compressed_kb}"
        # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===
        report = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, AnalystReport, self.budget_manager)
        return {"status": "SUCCESS", "data": report}

    def execute_final_synthesis(self, knowledge_base: dict, model_name: str, user_config: Dict) -> dict:
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        prompt = f"**КОНТЕКСТ ПРОЕКТА:** {main_goal}\n\n**ТВОЯ РОЛЬ:** Ведущий аналитик-стратег.\n**ТВОЯ ЗАДАЧА:** Превратить базу фактов в структурированный аналитический документ, который поможет достичь цели проекта. Заполни ВСЕ поля JSON-схемы.\n\n**БАЗА ЗНАНИЙ:**\n{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}"
        return invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, FinalAnalysisReport, self.budget_manager)

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
        artifact = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, FinancialModelArtifact, self.budget_manager)
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
        artifact = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, UserStoryArtifact, self.budget_manager)
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
        report = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, FinalReport, self.budget_manager)
        return report.get('markdown_content', '')


class ReviserAgent(BaseAgent):
    """
    Агент-критик, который оценивает полноту и релевантность собранной информации.
    """
    # === ИЗМЕНЕНИЕ НАЧАТО: Принимаем и храним компрессор ===
    def __init__(self, llm_client, budget_manager, context_compressor: ContextCompressor):
        super().__init__(llm_client, budget_manager)
        self.context_compressor = context_compressor
    # === ИЗМЕНЕНИЕ ОКОНЧЕНО ===

    def execute(self, task: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [ReviserAgent] -> Задача '{task['task_id']}' на модели {model_name}...")
        
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        knowledge_base = task.get("knowledge_base", {})
        remaining_tasks = task.get("remaining_tasks", [])

        # === ИЗМЕНЕНИЕ НАЧАТО: Используем компрессор ===
        task_desc = f"Провести ревизию собранной информации на предмет ее достаточности для достижения главной цели: {main_goal}"
        compressed_kb = self.context_compressor.compress(knowledge_base, task_desc)

        prompt = f"""
**ТВОЯ РОЛЬ:** Ведущий Исследователь-Стратег и внутренний критик.

**ГЛАВНАЯ ЦЕЛЬ ПРОЕКТА:**
{main_goal}

**СВОДКА УЖЕ СОБРАННЫХ ФАКТОВ (СЖАТЫЙ КОНТЕКСТ):**
{compressed_kb}

**ЗАДАЧИ, КОТОРЫЕ ЕЩЕ ОСТАЛОСЬ ВЫПОЛНИТЬ В ЭТОЙ ФАЗЕ:**
{json.dumps(remaining_tasks, ensure_ascii=False, indent=2)}

**ТВОЯ ЗАДАЧА - ПРОВЕСТИ РЕВИЗИЮ:**
1.  **Оцени Достаточность:** Достаточно ли информации в сводке для ответа на главный вопрос проекта?
2.  **Найди "Слепые Зоны":** Каких критически важных данных все еще не хватает?
3.  **Прими Решение:**
    - Если информация полна, установи `is_sufficient: true`.
    - Если нужны доработки, установи `is_sufficient: false`, дай `feedback` и предложи формулировки для новых задач в `new_task_suggestions`.

Верни результат в виде JSON, соответствующего схеме `RevisionReport`.
"""
        report = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, RevisionReport, self.budget_manager)
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
        outline = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-pro", "gemini-2.5-flash", prompt, ReportOutline, self.budget_manager)
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
        section = invoke_llm_for_json_with_retry(self.llm_client, "gemini-2.5-flash", "gemini-2.5-flash-lite", prompt, ReportSection, self.budget_manager)
        return section

