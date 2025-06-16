# -*- coding: utf-8 -*-
# 📄 Файл: __init__.py
# 📂 Путь: core/tools/__init__.py
# 📌 Назначение: Инициализация пакета утилит. Экспорт ключевых компонентов: эмбеддинг, NER, семантический поиск, генерация аннотаций

# ——— Эмбеддинги ———
from .embedder import EmbeddingService

# ——— Извлечение сущностей (NER) ———
from .extractor import extract_entities

# ——— Семантический поиск ———
from .semantic_search import SemanticSearch, semantic_search

# ——— Генерация аннотаций ———
from .summary_generator import SummaryGenerator, generate_summary

# Ниже — опционально, при наличии этих модулей
try:
    from .graph_tools import GraphTools
except ImportError:
    GraphTools = None

try:
    from .loader import FileLoader, SmartLoader
except ImportError:
    FileLoader = SmartLoader = None

try:
    from .async_tasks import celery_app, create_status_task
except ImportError:
    celery_app = create_status_task = None

__all__ = [
    "EmbeddingService",
    "extract_entities",
    "SemanticSearch", "semantic_search",
    "SummaryGenerator", "generate_summary",
    # опционально
    "FileLoader", "SmartLoader",
    "GraphTools",
    "celery_app", "create_status_task",
]
