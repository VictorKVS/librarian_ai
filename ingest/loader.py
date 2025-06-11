# 📄 Файл: loader.py
# 📂 Путь: ingest/
# 📌 Назначение: Определяет тип входного файла и извлекает из него сырой текст + метаданные. Принимает путь к файлу, возвращает текст и словарь с описанием источника.

import os


def load_file(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    metadata = {"type": ext[1:] if ext.startswith('.') else ext}

    if ext in ['.txt']:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), metadata

    elif ext in ['.md']:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read(), metadata

    elif ext in ['.docx']:
        from docx import Document
        doc = Document(filepath)
        text = "\n".join([p.text for p in doc.paragraphs])
        return text, metadata

    elif ext in ['.pdf']:
        from PyPDF2 import PdfReader
        reader = PdfReader(filepath)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text, metadata

    else:
        raise ValueError(f"❗ Неподдерживаемый формат файла: {ext}")
