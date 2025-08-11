import os
import re
from typing import List

# Несмотря на то, что requests и BeautifulSoup разрешены,
# для данной конкретной задачи (чтение локального файла-документа)
# они не требуются. Код должен быть максимально минималистичным
# и использовать только то, что необходимо для выполнения задачи.
# Импорты стандартных библиотек os и re являются достаточными.

def read_internal_document(document_id: str) -> str:
    """
    Читает полное текстовое содержимое внутреннего документа или результата
    предыдущей задачи из защищенного хранилища.

    Эта функция предназначена для доступа к файлам в строго определенной
    директории, которая считается "внутренним хранилищем". Путь к этой
    директории определяется переменной окружения `INTERNAL_DOCS_PATH`.
    Если переменная не установлена, используется относительный путь './internal_documents/'.

    В целях безопасности функция выполняет следующие проверки:
    1. Запрещает использование '..' в `document_id` для предотвращения выхода
       за пределы рабочей директории.
    2. Нормализует путь и убеждается, что итоговый путь к файлу находится
       внутри разрешенной директории хранилища.

    Args:
        document_id (str): Идентификатор или имя документа для чтения.
                           Предполагается, что это имя файла, возможно,
                           без расширения (будет предпринята попытка добавить .txt).
                           Например, 'data_04_tech_deep_dive'.

    Returns:
        str: Полное текстовое содержимое запрошенного документа.

    Raises:
        Exception: Если `document_id` содержит недопустимые символы,
                   если документ не найден, если происходит попытка доступа
                   за пределы разрешенной директории, или в случае других
                   ошибок чтения файла.
    """
    try:
        # 1. Определение базовой директории для документов
        base_path = os.environ.get("INTERNAL_DOCS_PATH", "internal_documents")
        
        # Создаем директорию, если она не существует, для удобства первого запуска
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        # 2. Проверка безопасности: предотвращение path traversal атак
        if ".." in document_id or "/" in document_id or "\\" in document_id:
            raise Exception(f"Ошибка безопасности: document_id '{document_id}' содержит недопустимые символы ('..', '/', '\\').")

        # 3. Попытка найти файл с расширением .txt или без него
        possible_filenames = [f"{document_id}.txt", document_id]
        file_path_to_read = None

        for filename in possible_filenames:
            potential_path = os.path.join(base_path, filename)
            
            # 4. Дополнительная проверка безопасности: убедиться, что мы остаемся в base_path
            abs_base_path = os.path.realpath(base_path)
            abs_potential_path = os.path.realpath(potential_path)

            if not abs_potential_path.startswith(abs_base_path):
                raise Exception(f"Ошибка безопасности: Попытка доступа к файлу '{document_id}' за пределами разрешенной директории.")

            if os.path.isfile(potential_path):
                file_path_to_read = potential_path
                break
        
        if not file_path_to_read:
            raise FileNotFoundError(f"Документ с ID '{document_id}' не найден в хранилище '{base_path}'.")

        # 5. Чтение и возврат содержимого файла
        with open(file_path_to_read, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content

    except FileNotFoundError as e:
        raise Exception(f"Не удалось найти документ: {e}")
    except Exception as e:
        # Перехватываем все остальные исключения (включая наши собственные ошибки безопасности)
        # и перевыбрасываем их в стандартизированном формате.
        raise Exception(f"Произошла ошибка при чтении документа '{document_id}': {e}")