# core/tool_registry.py
import os
import importlib.util
import inspect
from typing import Dict, Any, Callable, List

class ToolRegistry:
    """
    Центральный реестр для хранения, управления и безопасного выполнения инструментов.
    Отвечает за регистрацию как встроенных, так и динамически сгенерированных инструментов.
    """
    def __init__(self, generated_tools_dir: str = "tools/generated"):
        self.generated_tools_dir = generated_tools_dir
        os.makedirs(self.generated_tools_dir, exist_ok=True)
        # Создаем __init__.py, чтобы директория была импортируемым пакетом
        init_path = os.path.join(self.generated_tools_dir, "__init__.py")
        if not os.path.exists(init_path):
            with open(init_path, "w") as f:
                f.write("# This file makes tools/generated a Python package.\n")

        self.tools: Dict[str, Callable] = {}
        self._load_builtin_tools()
        self._load_generated_tools()
        print(f"-> ToolRegistry инициализирован. Загружено инструментов: {len(self.tools)}")

    def _load_builtin_tools(self):
        """Загружает предопределенные, надежные инструменты."""
        try:
            from tools.web_search import perform_search
            self.tools['web_search'] = perform_search
            print("   [ToolRegistry] Встроенный инструмент 'web_search' успешно загружен.")
        except ImportError as e:
            print(f"!!! [ToolRegistry] ВНИМАНИЕ: Не удалось загрузить встроенные инструменты. Ошибка: {e}")

    def _load_generated_tools(self):
        """Загружает ранее сгенерированные инструменты при старте системы."""
        print("   [ToolRegistry] Поиск ранее сгенерированных инструментов...")
        for filename in os.listdir(self.generated_tools_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                tool_name = filename[:-3]
                try:
                    self._import_and_register(tool_name)
                    print(f"   [ToolRegistry] <- Загружен инструмент '{tool_name}' из кэша.")
                except Exception as e:
                    print(f"!!! [ToolRegistry] Ошибка загрузки инструмента '{tool_name}': {e}")

    def _import_and_register(self, tool_name: str):
        """Безопасно импортирует и регистрирует функцию инструмента из файла."""
        module_path = os.path.join(self.generated_tools_dir, f"{tool_name}.py")
        
        # Динамически импортируем модуль из файла
        spec = importlib.util.spec_from_file_location(f"tools.generated.{tool_name}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Получаем функцию из модуля (предполагаем, что имя функции совпадает с именем файла/инструмента)
        tool_function = getattr(module, tool_name)
        self.tools[tool_name] = tool_function

    def register_tool(self, name: str, code: str):
        """
        Сохраняет код инструмента в файл и регистрирует его в системе.
        Это основной метод, вызываемый после генерации кода ToolSmithAgent'ом.
        """
        if not name.isidentifier():
            raise ValueError(f"Имя инструмента '{name}' не является валидным идентификатором Python.")
        
        file_path = os.path.join(self.generated_tools_dir, f"{name}.py")
        
        print(f"   [ToolRegistry] -> Регистрирую новый инструмент '{name}'...")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            self._import_and_register(name)
            print(f"   [ToolRegistry] <- Инструмент '{name}' успешно зарегистрирован и готов к использованию.")
        except Exception as e:
            print(f"!!! [ToolRegistry] КРИТИЧЕСКАЯ ОШИБКА регистрации инструмента '{name}': {e}")
            # В случае ошибки удаляем некорректный файл
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

    def get_tools_for_prompt(self) -> str:
        """
        Генерирует текстовое описание всех доступных инструментов для вставки в промпт LLM.
        """
        if not self.tools:
            return "Инструменты не доступны."
        
        descriptions = ["СПИСОК ДОСТУПНЫХ ИНСТРУМЕНТОВ:"]
        for name, func in self.tools.items():
            signature = inspect.signature(func)
            docstring = inspect.getdoc(func) or "Описание отсутствует."
            descriptions.append(f"- Инструмент: `{name}{signature}`\n  Описание: {docstring}")
        
        return "\n".join(descriptions)

    def use_tool(self, name: str, args: Dict[str, Any], state: Dict) -> Any:
        """
        Выполняет указанный инструмент и обновляет состояние графа (например, visited_urls).
        """
        if name not in self.tools:
            return f"Ошибка: Инструмент '{name}' не найден."
        
        tool_func = self.tools[name]
        print(f"   [ToolRegistry] -> Выполняю инструмент '{name}' с аргументами: {args}")
        try:
            # --- ЛОГИКА ОБНОВЛЕНИЯ СОСТОЯНИЯ ---
            # Если это поисковый инструмент, мы можем извлечь URL из его результата
            if name == 'web_search':
                # Сначала проверяем, нет ли URL уже в аргументах (например, для чтения конкретной страницы)
                url_to_check = args.get('url') or args.get('link')
                if url_to_check and url_to_check in state.get('visited_urls', []):
                    print(f"   [ToolRegistry] <- URL '{url_to_check}' уже посещался. Пропускаю.")
                    return "Информация с этого URL уже была проанализирована ранее."

            result = tool_func(**args)

            # После выполнения извлекаем URL из результата, если возможно
            if isinstance(result, dict) and 'items' in result:
                urls_found = [item.get('link') for item in result['items'] if item.get('link')]
                if urls_found:
                    state.setdefault('visited_urls', []).extend(urls_found)
                    print(f"   [ToolRegistry] Добавлено {len(urls_found)} URL в список посещенных.")
            
            print(f"   [ToolRegistry] <- Инструмент '{name}' успешно выполнен.")
            return result
        except Exception as e:
            print(f"!!! [ToolRegistry] Ошибка выполнения инструмента '{name}': {e}")
            return f"Ошибка при выполнении инструмента '{name}': {e}"