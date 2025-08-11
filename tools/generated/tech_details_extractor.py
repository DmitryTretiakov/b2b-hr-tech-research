import re
from typing import Dict, Any, List, Set

def tech_details_extractor(text_content: str) -> Dict[str, Any]:
    """
    Анализирует предоставленный текстовый контент для извлечения структурированных технических данных.

    Эта функция использует комбинацию поиска по ключевым словам и регулярных выражений для
    идентификации и извлечения информации по следующим категориям:
    - Технологический стек (языки, фреймворки, базы данных и т.д.)
    - Детали API (тип, аутентификация, эндпоинты, документация)
    - Версии программного обеспечения
    - Описанные ограничения системы

    Функция разработана для работы в изолированной среде с ограниченными зависимостями
    и не выполняет никаких внешних сетевых запросов. Вся обработка происходит
    исключительно на основе входной строки `text_content`.

    Args:
        text_content: Строка, содержащая необработанный текст для анализа.
                      Это может быть содержимое веб-страницы, документации или любого другого
                      технического документа.

    Returns:
        Словарь, содержащий извлеченные структурированные данные. Структура словаря:
        {
            "tech_stack": List[str],
            "api_details": {
                "has_api": bool,
                "type": List[str],
                "endpoints": List[str],
                "authentication": List[str],
                "documentation_url": List[str]
            },
            "versions": List[str],
            "limitations": List[str]
        }

    Raises:
        Exception: Если `text_content` не является строкой или пуст, или если в процессе
                   анализа возникает непредвиденная ошибка.
    """
    if not isinstance(text_content, str) or not text_content:
        raise Exception("Входной параметр 'text_content' должен быть непустой строкой.")

    try:
        # --- Инициализация структур для хранения результатов ---
        results: Dict[str, Any] = {
            "tech_stack": [],
            "api_details": {
                "has_api": False,
                "type": [],
                "endpoints": [],
                "authentication": [],
                "documentation_url": []
            },
            "versions": [],
            "limitations": []
        }

        # --- База знаний: Ключевые слова для поиска технологий ---
        # Ключи - канонические названия, значения - варианты для поиска (в нижнем регистре)
        TECH_KEYWORDS = {
            # Языки программирования
            "Python": ["python"], "JavaScript": ["javascript", " js ", "es6", "es2015"], "TypeScript": ["typescript", " ts "],
            "Java": ["java "], "Kotlin": ["kotlin"], "Go": ["golang", " go "], "Rust": ["rust"], "PHP": ["php"],
            "C#": ["c#", "csharp"], "C++": ["c++", "cpp"], "Ruby": ["ruby"], "Swift": ["swift"], "Scala": ["scala"],
            # Фреймворки и библиотеки
            "React": ["react.js", "reactjs", "react"], "Angular": ["angular.js", "angularjs", "angular"],
            "Vue.js": ["vue.js", "vuejs", "vue"], "Node.js": ["node.js", "nodejs"], "Express.js": ["express.js", "expressjs"],
            "Django": ["django"], "Flask": ["flask"], "FastAPI": ["fastapi"], "Spring": ["spring boot", "spring framework"],
            "Ruby on Rails": ["ruby on rails", "rails"], ".NET": [".net core", ".net framework", " asp.net"],
            "jQuery": ["jquery"], "TensorFlow": ["tensorflow"], "PyTorch": ["pytorch"], "Scikit-learn": ["scikit-learn", "sklearn"],
            # Базы данных
            "PostgreSQL": ["postgresql", "postgres"], "MySQL": ["mysql"], "MariaDB": ["mariadb"],
            "SQLite": ["sqlite"], "MongoDB": ["mongodb"], "Redis": ["redis"], "Cassandra": ["cassandra"],
            "Elasticsearch": ["elasticsearch"], "Microsoft SQL Server": ["ms sql", "sql server"], "Oracle": ["oracle database"],
            # Облачные платформы и DevOps
            "AWS": ["aws", "amazon web services"], "Google Cloud": ["gcp", "google cloud"], "Microsoft Azure": ["azure"],
            "Docker": ["docker"], "Kubernetes": ["kubernetes", "k8s"], "Terraform": ["terraform"], "Ansible": ["ansible"],
            "Jenkins": ["jenkins"], "Git": [" git "],
            # Веб-серверы
            "Nginx": ["nginx"], "Apache": ["apache http server", "httpd"], "IIS": ["iis", "internet information services"]
        }

        lower_text = text_content.lower()
        
        # 1. Извлечение технологического стека
        found_stack: Set[str] = set()
        for tech, keywords in TECH_KEYWORDS.items():
            for keyword in keywords:
                # Используем `\b` (границу слова) для более точного поиска, где это возможно
                # Для ключевых слов с пробелами или спецсимволами простой поиск `in` надежнее
                if ' ' in keyword or '.' in keyword or '#' in keyword:
                    if keyword in lower_text:
                        found_stack.add(tech)
                        break
                else:
                    if re.search(r'\b' + re.escape(keyword) + r'\b', lower_text):
                        found_stack.add(tech)
                        break
        results["tech_stack"] = sorted(list(found_stack))

        # 2. Извлечение деталей API
        api_keywords = ["api", " a p i "]
        if any(keyword in lower_text for keyword in api_keywords):
            results["api_details"]["has_api"] = True

            # Типы API
            api_types = {"REST": ["restful", "rest api"], "GraphQL": ["graphql"], "SOAP": ["soap"]}
            found_types: Set[str] = set()
            for api_type, keywords in api_types.items():
                if any(keyword in lower_text for keyword in keywords):
                    found_types.add(api_type)
            results["api_details"]["type"] = sorted(list(found_types))

            # Аутентификация
            auth_methods = {
                "API Key": ["api key", "api-key"], "OAuth": ["oauth"], "JWT": ["jwt", "json web token"],
                "Bearer Token": ["bearer token"], "Basic Auth": ["basic authentication"]
            }
            found_auth: Set[str] = set()
            for auth_method, keywords in auth_methods.items():
                if any(keyword in lower_text for keyword in keywords):
                    found_auth.add(auth_method)
            results["api_details"]["authentication"] = sorted(list(found_auth))

            # Эндпоинты (простой эвристический поиск)
            # Ищем пути, которые выглядят как эндпоинты, особенно в блоках кода или после слов GET/POST/PUT/DELETE
            endpoint_pattern = r'(?:GET|POST|PUT|DELETE|PATCH)\s+([/\w\d\-\._~:?#\[\]@!$&\'()*+,;=%]+)|`(/[\w\d\-\._~:?#\[\]@!$&\'()*+,;=%]+)`'
            found_endpoints: Set[str] = set()
            for match in re.finditer(endpoint_pattern, text_content, re.IGNORECASE):
                # match.group(1) для GET/POST, match.group(2) для путей в `...`
                endpoint = match.group(1) or match.group(2)
                if endpoint and endpoint.startswith('/'):
                    found_endpoints.add(endpoint.strip())
            results["api_details"]["endpoints"] = sorted(list(found_endpoints))

            # Ссылки на документацию
            doc_urls: Set[str] = set()
            # Ищем URL-адреса, находящиеся рядом со словами "documentation", "docs", "reference"
            doc_pattern = r'(?:documentation|docs|reference|guide)[\s\w]*?[:\s]*?(https?://[\w\d\-\./_?=&%]+)'
            for match in re.finditer(doc_pattern, lower_text):
                doc_urls.add(match.group(1))
            results["api_details"]["documentation_url"] = sorted(list(doc_urls))


        # 3. Извлечение версий
        # Ищем паттерны типа v1.2.3, version 1.2, или название технологии + номер версии
        version_pattern = r'\b(?:v|version\s?|ver\.\s?)(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9\.]+)?)\b'
        found_versions: Set[str] = set(re.findall(version_pattern, text_content, re.IGNORECASE))
        
        # Поиск версий рядом с уже найденными технологиями
        for tech in results["tech_stack"]:
            # Создаем паттерн для поиска, например, "Python 3.9"
            tech_version_pattern = re.escape(tech) + r'\s+(\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9\.]+)?)\b'
            matches = re.findall(tech_version_pattern, text_content, re.IGNORECASE)
            for version in matches:
                found_versions.add(f"{tech} {version}")
        results["versions"] = sorted(list(found_versions))

        # 4. Извлечение ограничений
        # Ищем предложения, содержащие ключевые слова, связанные с ограничениями
        limitations_keywords = [
            "limitation", "limitations", "restriction", "restrictions", "constraint", "constraints",
            "known issue", "known issues", "bottleneck", "incompatible", "not compatible",
            "scalability concern", "performance degradation"
        ]
        found_limitations: Set[str] = set()
        # Разбиваем текст на предложения для контекстного анализа
        sentences = re.split(r'(?<=[.!?])\s+', text_content)
        for sentence in sentences:
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in limitations_keywords):
                # Добавляем чистое предложение без лишних пробелов
                found_limitations.add(sentence.strip())
        results["limitations"] = sorted(list(found_limitations))

        return results

    except Exception as e:
        # Перехватываем любые непредвиденные ошибки во время анализа и оборачиваем их
        # в стандартное исключение инструмента.
        raise Exception(f"Произошла ошибка при анализе текста: {e}")