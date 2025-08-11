import json
import re
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, List, Union

def financial_modeling_tool(query: str) -> Dict[str, Any]:
    """
    Выполняет финансовое моделирование на основе структурированного запроса в формате JSON.

    Инструмент создает финансовые прогнозы (например, отчет о прибылях и убытках) на заданное количество периодов.
    Он использует набор предположений, начальные значения и формулы для расчета каждой строки модели.
    Инструмент также может извлекать числовые данные с веб-страниц для использования в расчетах.

    Структура JSON-запроса (`query`):
    {
        "model_name": "Название вашей модели (опционально)",
        "periods": 5, // (Обязательно) Количество периодов (например, лет) для прогнозирования.
        "start_year": 2024, // (Опционально) Год начала для заголовков таблицы.
        "assumptions": { // (Опционально) Словарь с ключевыми предположениями.
            "revenue_growth_rate": 0.15,
            "tax_rate": 0.25
        },
        "line_items": [ // (Обязательно) Список строк модели для расчета.
            {
                "name": "Revenue", // Название строки.
                "base_value": 500000, // Начальное значение для первого периода (t=0).
                "formula": "prev('Revenue') * (1 + assumptions['revenue_growth_rate'])" // Формула для t > 0.
            },
            {
                "name": "Net Income",
                "formula": "self('EBIT') - self('Taxes')" // Формула, используемая для всех периодов.
            }
        ]
    }

    Синтаксис формул:
    - `self('Item Name')`: Возвращает значение другой строки ('Item Name') за ТЕКУЩИЙ период.
      Порядок строк в `line_items` важен для зависимостей.
    - `prev('Item Name')`: Возвращает значение строки ('Item Name') за ПРЕДЫДУЩИЙ период.
      Нельзя использовать в первом периоде (t=0); для этого используйте `base_value`.
    - `assumptions['key']`: Возвращает значение из словаря `assumptions`.
    - `t`: Индекс текущего периода (начинается с 0).
    - `max(a, b, ...)`: Возвращает максимальное из значений.
    - `min(a, b, ...)`: Возвращает минимальное из значений.
    - `scrape_value('url', 'css_selector')`: Загружает URL, находит элемент по CSS-селектору,
      извлекает из него числовое значение. Аргументы должны быть строковыми литералами в кавычках.
    - Стандартные математические операторы: `+`, `-`, `*`, `/` и скобки `()`.

    Args:
        query (str): Строка в формате JSON, описывающая финансовую модель.

    Returns:
        Dict[str, Any]: Словарь, содержащий статус выполнения и результаты моделирования
                        в виде таблицы.

    Raises:
        Exception: Если происходит ошибка при парсинге JSON, выполнении расчетов,
                   сетевом запросе или если структура запроса некорректна.
    """

    # --- Вспомогательные функции, определенные внутри основной функции для инкапсуляции ---

    def _evaluate_math(expression: str) -> float:
        """Безопасно вычисляет математическое выражение с помощью алгоритма сортировочной станции."""
        # Настройка операторов
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}

        def apply_op(operators, values):
            op = operators.pop()
            right = values.pop()
            left = values.pop()
            if op == '+': values.append(left + right)
            elif op == '-': values.append(left - right)
            elif op == '*': values.append(left * right)
            elif op == '/':
                if right == 0: raise ValueError("Division by zero")
                values.append(left / right)

        tokens = re.findall(r"(-?\d+\.?\d*|[+\-*/()])", expression.replace(" ", ""))
        values_stack = []
        ops_stack = []

        i = 0
        while i < len(tokens):
            token = tokens[i]
            # Обработка унарного минуса
            if token == '-' and (i == 0 or tokens[i-1] in '(+-*/'):
                # Это унарный минус, объединяем его со следующим числом
                if i + 1 < len(tokens) and re.match(r"\d+\.?\d*", tokens[i+1]):
                    values_stack.append(float(token + tokens[i+1]))
                    i += 1 # Пропускаем следующее число, так как оно уже обработано
                else:
                    raise ValueError(f"Invalid expression: misplaced unary minus in '{expression}'")
            elif re.match(r"-?\d+\.?\d*", token):
                values_stack.append(float(token))
            elif token == '(':
                ops_stack.append(token)
            elif token == ')':
                while ops_stack and ops_stack[-1] != '(':
                    apply_op(ops_stack, values_stack)
                if not ops_stack or ops_stack.pop() != '(':
                    raise ValueError(f"Mismatched parentheses in '{expression}'")
            elif token in precedence:
                while ops_stack and ops_stack[-1] in precedence and precedence[ops_stack[-1]] >= precedence[token]:
                    apply_op(ops_stack, values_stack)
                ops_stack.append(token)
            else:
                raise ValueError(f"Unknown token '{token}' in expression '{expression}'")
            i += 1

        while ops_stack:
            apply_op(ops_stack, values_stack)

        if len(values_stack) != 1:
            raise ValueError(f"Invalid mathematical expression: '{expression}'")
        return values_stack[0]

    def _scrape_value(url: str, selector: str) -> float:
        """Извлекает числовое значение с веб-страницы."""
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            element = soup.select_one(selector)

            if not element:
                raise ValueError(f"CSS selector '{selector}' not found on page {url}")

            text_value = element.get_text(strip=True)
            # Лучшая попытка очистить текст до числа
            cleaned_text = re.sub(r'[^\d.-]', '', text_value)
            if not cleaned_text or cleaned_text == '.':
                raise ValueError(f"Found element with selector '{selector}', but it contained no numeric data (raw text: '{text_value}')")

            return float(cleaned_text)
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error while scraping {url}: {e}")
        except Exception as e:
            raise Exception(f"Failed to scrape and parse value from {url} with selector '{selector}': {e}")

    def _evaluate_formula(formula: str, context: Dict[str, Any]) -> float:
        """Рекурсивно вычисляет формулу, разрешая ссылки и функции."""
        MAX_LOOPS = 100
        for _ in range(MAX_LOOPS):
            # Если формула уже является числом, возвращаем его
            try:
                return float(formula)
            except (ValueError, TypeError):
                pass

            original_formula = formula

            # 1. Вычисляем самые вложенные функции: max(), min(), scrape_value()
            pattern = re.compile(r"\b(max|min|scrape_value)\(([^()]+)\)")
            match = pattern.search(formula)
            if match:
                func_name, args_str = match.groups()
                
                if func_name == 'scrape_value':
                    # Аргументы для scrape_value должны быть строковыми литералами
                    args = [arg.strip().strip("'\"") for arg in args_str.split(',', 1)]
                    if len(args) != 2: raise ValueError("scrape_value requires 2 arguments: url and selector")
                    result = _scrape_value(args[0], args[1])
                else: # max, min
                    # Вычисляем каждый аргумент, так как они могут быть выражениями
                    evaluated_args = [_evaluate_formula(str(arg).strip(), context) for arg in args_str.split(',')]
                    if func_name == 'max': result = max(evaluated_args)
                    elif func_name == 'min': result = min(evaluated_args)
                
                formula = formula.replace(match.group(0), str(result), 1)
                continue

            # 2. Заменяем ссылки: assumptions, self, prev, t
            temp_formula = formula
            temp_formula = re.sub(r"assumptions\['([^']+)']", lambda m: str(context['assumptions'][m.group(1)]), temp_formula)
            temp_formula = re.sub(r"self\('([^']+)'\)", lambda m: str(context['current_results'][m.group(1)]), temp_formula)
            temp_formula = re.sub(r"prev\('([^']+)'\)", lambda m: str(context['previous_results'][m.group(1)]), temp_formula)
            temp_formula = re.sub(r'\bt\b', str(context['t']), temp_formula)

            if temp_formula != formula:
                formula = temp_formula
                continue
            
            # 3. Если больше нет замен, вычисляем оставшееся математическое выражение
            try:
                return _evaluate_math(formula)
            except ValueError as e:
                raise ValueError(f"Could not evaluate expression '{formula}': {e}")

        raise ValueError(f"Formula evaluation timed out or has a circular dependency: {original_formula}")

    # --- Основная логика инструмента ---
    try:
        # 1. Парсинг и валидация JSON-запроса
        try:
            model_spec = json.loads(query)
        except json.JSONDecodeError as e:
            raise Exception(f"Invalid JSON in query: {e}")

        required_keys = ['periods', 'line_items']
        if not all(key in model_spec for key in required_keys):
            raise Exception(f"Query is missing one of the required keys: {required_keys}")

        # 2. Инициализация модели
        periods = int(model_spec['periods'])
        line_items_spec = model_spec['line_items']
        assumptions = model_spec.get('assumptions', {})
        start_year = model_spec.get('start_year')

        line_item_order = [item['name'] for item in line_items_spec]
        formulas = {item['name']: item for item in line_items_spec}
        results = {name: [0.0] * periods for name in line_item_order}

        # 3. Цикл расчетов по периодам
        for t in range(periods):
            current_period_results = {}
            previous_period_results = {name: results[name][t - 1] for name in line_item_order} if t > 0 else {}

            for name in line_item_order:
                spec = formulas[name]
                value = 0.0

                # Определение значения для текущего периода
                if t == 0 and 'base_value' in spec:
                    value = float(spec['base_value'])
                elif 'formula' in spec:
                    context = {
                        't': t,
                        'assumptions': assumptions,
                        'current_results': current_period_results,
                        'previous_results': previous_period_results,
                    }
                    # Проверка на prev() в первом периоде перед вычислением
                    if t == 0 and "prev(" in spec['formula']:
                         raise Exception(f"Line item '{name}' cannot use prev() in the first period (t=0). Provide a 'base_value'.")
                    value = _evaluate_formula(spec['formula'], context)
                elif t > 0 and 'base_value' in spec and 'formula' not in spec:
                    # Если формула не указана, переносим значение base_value (для фиксированных значений)
                    value = float(spec['base_value'])
                else:
                    raise Exception(f"Line item '{name}' has no 'formula' or 'base_value' applicable for period {t}.")

                results[name][t] = value
                current_period_results[name] = value

        # 4. Форматирование выходных данных
        headers = ["Line Item"] + ([str(start_year + i) for i in range(periods)] if start_year else [f"Period {i+1}" for i in range(periods)])
        rows = [[name] + [round(val, 2) for val in results[name]] for name in line_item_order]

        output_data = {
            "model_name": model_spec.get("model_name", "Financial Model"),
            "assumptions": assumptions,
            "results_table": {
                "headers": headers,
                "rows": rows
            }
        }

        return {"status": "success", "data": output_data}

    except Exception as e:
        # Перехватываем все ожидаемые и неожиданные ошибки и перевыбрасываем их
        # с понятным сообщением, чтобы соответствовать требованиям.
        raise Exception(f"Financial modeling tool failed: {e}")