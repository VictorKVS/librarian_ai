# -*- coding: utf-8 -*-
# 📄 Файл: archive_extractors.py
# 📂 Путь: core/tools/archive_extractors.py
# 📌 Назначение: Извлечение текста из ZIP-архивов с поддержкой декодирования и фильтрации

import zipfile
import os
import logging

logger = logging.getLogger(__name__)

def extract_text_from_zip(zip_path: str, extract_to: str = "./tmp") -> list[str]:
    """
    Извлекает текстовые файлы из ZIP-архива.

    Args:
        zip_path (str): Путь к архиву.
        extract_to (str): Папка, куда извлекаются файлы.

    Returns:
        list[str]: Список текстов, извлечённых из файлов.
    """
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} не является корректным ZIP-архивом")

    os.makedirs(extract_to, exist_ok=True)
    texts = []

    with zipfile.ZipFile(zip_path, 'r') as archive:
        for filename in archive.namelist():
            if not filename.endswith(('.txt', '.md', '.log', '.csv')):
                logger.debug(f"Пропущен файл: {filename}")
                continue
            archive.extract(filename, path=extract_to)
            full_path = os.path.join(extract_to, filename)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
            except UnicodeDecodeError:
                logger.warning(f"Ошибка декодирования UTF-8: {filename}")
            except Exception as e:
                logger.error(f"Ошибка при чтении {filename}: {e}")

    return texts
