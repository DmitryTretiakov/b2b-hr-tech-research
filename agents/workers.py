# agents/workers.py
import json
from datetime import datetime, timezone
from typing import Dict
from agents.base_agent import BaseAgent
from utils.helpers import invoke_llm_for_json_with_retry
from agents.models import (
    FactExtractionReport, BatchQualityAssessmentReport, AnalystReport, 
    FinalReport, FinalAnalysisReport, SanityCheckReport
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

        # Формируем высокоуровневый контекст для агента
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        high_level_context = f"**КОНТЕКСТ ВСЕГО ПРОЕКТА:**\nТы работаешь над достижением следующей главной цели: '{main_goal}'. Твоя текущая задача - это один из шагов на пути к этой цели. Выполняй ее, держа в уме конечный результат."

        available_tools = self.tool_registry.get_tools_for_prompt()
        initial_prompt = f"""{self.role_prompt}
{high_level_context}

**ТВОЯ ТЕКУЩАЯ ЗАДАЧА:** '{task['description']}'

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
                    tool_result = self.tool_registry.use_tool(tool_call.get("tool_name"), tool_call.get("args", {}))
                    observation = f"OBSERVATION:\n```\n{str(tool_result)[:3000]}\n```" # Обрезаем для экономии токенов
                    conversation_history.append(observation)
                    print(f"      [ReAct] Инструмент '{tool_call.get('tool_name')}' выполнен.")
                elif "finish" in action_data:
                    print("      [ReAct] Агент решил завершить сбор информации.")
                    break
                else: 
                    raise ValueError("Неверный формат JSON-действия.")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"      [ReAct] !!! Ошибка обработки ответа LLM: {e}. Прошу исправиться.")
                conversation_history.append(f"OBSERVATION: Ошибка обработки ответа: {e}. Пожалуйста, верни JSON с ключом 'tool_to_use' или 'finish'.")

        print("   [ReAct] -> Перехожу к финальному синтезу фактов...")
        final_synthesis_prompt = f"{self.role_prompt}\nПроанализируй всю переписку и извлеки 3-5 ключевых фактов. Заполни все поля.\n\n**ИСТОРИЯ РАБОТЫ:**\n{''.join(conversation_history)}"
        
        # Для синтеза используем более мощную модель
        synthesis_model = "gemini-2.5-flash"
        report = invoke_llm_for_json_with_retry(self.llm_client, synthesis_model, "gemini-2.5-flash-lite", final_synthesis_prompt, FactExtractionReport, self.budget_manager)

        if not report or 'extracted_facts' not in report: 
            return []
        
        for fact in report['extracted_facts']: 
            fact['created_at'] = datetime.now(timezone.utc).isoformat()
        
        print(f"   [{self.__class__.__name__}] <- Задача '{task['task_id']}' выполнена. Извлечено {len(report['extracted_facts'])} фактов.")
        return report['extracted_facts']

class ResearcherAgent(BaseResearchAgent):
    role_prompt: str = "Твоя роль: Ассистент-исследователь. Твоя цель — найти подтверждающие, основные факты по задаче."

class ContrarianAgent(BaseResearchAgent):
    role_prompt: str = "Твоя роль: 'Адвокат Дьявола'. Твоя цель — найти опровержения, критику и провальные кейсы по задаче."

class QualityAssessorAgent(BaseAgent):
    """Оценивает качество фактов и возвращает отчет."""
    def execute(self, facts_to_assess: list, model_name: str, user_config: Dict) -> dict:
        print(f"   [QualityAssessorAgent] -> Оцениваю {len(facts_to_assess)} фактов на модели {model_name}...")
        prompt = f"Твоя роль: Контролер качества. Для КАЖДОГО факта вынеси вердикт по чек-листу (Конкретность, Доказуемость, Полнота) и верни JSON-отчет.\nФАКТЫ:\n{json.dumps(facts_to_assess, ensure_ascii=False, indent=2)}"
        return invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash-lite", prompt,
            BatchQualityAssessmentReport, self.budget_manager
        )

class FixerAgent(BaseAgent):
    """Исправляет некачественные факты и возвращает список исправленных."""
    def execute(self, facts_to_fix: list, model_name: str, user_config: Dict) -> list:
        print(f"   [FixerAgent] -> Исправляю {len(facts_to_fix)} фактов на модели {model_name}...")
        prompt = f"Твоя роль: Редактор. Исправь факты на основе `feedback`, сохранив `claim_id`. Если исправить невозможно, не включай факт в ответ.\nФАКТЫ:\n{json.dumps(facts_to_fix, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            FactExtractionReport, self.budget_manager
        )
        return report.get('extracted_facts', [])

class SanityCheckCritic(BaseAgent):
    """Проводит финальную проверку на здравый смысл."""
    def execute(self, facts_to_check: list, model_name: str, user_config: Dict) -> list:
        print(f"   [SanityCheckCritic] -> Финальная проверка {len(facts_to_check)} фактов на модели {model_name}...")
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        prompt = f"**Главная цель проекта:** {main_goal}\n\nТвоя роль: Старший аналитик. Проверь факты на коммерческую релевантность для достижения главной цели и отсутствие 'воды'. Верни JSON со списком `verified_claim_ids` тех, кто прошел проверку.\nФАКТЫ:\n{json.dumps(facts_to_check, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            SanityCheckReport, self.budget_manager
        )
        verified_ids = set(report.get('verified_claim_ids', []))
        return [fact for fact in facts_to_check if fact['claim_id'] in verified_ids]

class AnalystAgent(BaseAgent):
    """Синтезирует инсайты из Базы Знаний."""
    def execute_reflection(self, knowledge_base: dict, model_name: str, user_config: Dict) -> dict:
        print(f"   [AnalystAgent] -> Синтезирую инсайты для рефлексии на модели {model_name}...")
        prompt = f"Твоя роль: Старший аналитик. Проанализируй Базу Знаний и предоставь краткую сводку для планировщика: 3-5 ключевых инсайтов и 2-3 пробела в данных.\nБАЗА ЗНАНИЙ:\n{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}"
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            AnalystReport, self.budget_manager
        )
        return {"status": "SUCCESS", "data": report}

    def execute_final_synthesis(self, knowledge_base: dict, model_name: str, user_config: Dict) -> dict:
        """Создает финальную структурированную сводку по всей Базе Знаний."""
        print(f"   [AnalystAgent] -> Выполняю финальный синтез всей Базы Знаний на модели {model_name}...")
        main_goal = user_config.get("user_context", {}).get("main_goal", "Цель не определена.")
        prompt = f"""
**КОНТЕКСТ ПРОЕКТА:** {main_goal}

**ТВОЯ РОЛЬ:** Ведущий аналитик-стратег.
**ТВОЯ ЗАДАЧА:** Превратить разрозненную базу фактов в структурированный аналитический документ, который поможет достичь цели проекта. Проанализируй ВСЮ базу знаний и заполни ВСЕ поля JSON-схемы.

**БАЗА ЗНАНИЙ ДЛЯ АНАЛИЗА:**
---
{json.dumps(knowledge_base, ensure_ascii=False, indent=2)}
---
"""
        return invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash", prompt,
            FinalAnalysisReport, self.budget_manager
        )

class ReportWriterAgent(BaseAgent):
    """Пишет отчет по структурированным данным."""
    def execute(self, analysis_data: dict, model_name: str, user_config: Dict) -> str:
        """Принимает структурированные данные и возвращает готовый Markdown-текст."""
        print(f"   [ReportWriterAgent] -> Пишу отчет на модели {model_name}...")
        profile = user_config.get("user_context", {}).get("profile", "Профиль не определен.")
        prompt = f"""
**КОНТЕКСТ АУДИТОРИИ:** {profile}

**ТВОЯ РОЛЬ:** Профессиональный копирайтер и редактор.
**ТВОЯ ЗАДАЧА:** Превратить структурированные аналитические данные в связный, хорошо читаемый и профессионально оформленный Markdown-отчет, адаптированный под указанную аудиторию.

**СТРОГИЕ ИНСТРУКЦИИ:**
1.  **Следуй структуре:** Используй все предоставленные поля (`title`, `executive_summary` и т.д.).
2.  **Вставляй маркеры цитирования:** После КАЖДОГО утверждения, которое основано на данных из `key_findings`, ты ДОЛЖЕН вставить специальный маркер `[CITE:claim_id]`.
3.  **Не выдумывай:** Не добавляй информацию, которой нет в предоставленных данных.
4.  **Форматирование:** Используй Markdown для заголовков, списков и выделения текста.

**ДАННЫЕ ОТ АНАЛИТИКА:**
---
{json.dumps(analysis_data, ensure_ascii=False, indent=2)}
---
"""
        report = invoke_llm_for_json_with_retry(
            self.llm_client, model_name, "gemini-2.5-flash-lite", prompt,
            FinalReport, self.budget_manager
        )
        return report.get('markdown_content', '')