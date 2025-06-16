# 📄 Файл: async_tasks.py
# 📂 Путь: core/tasks/
# 📌 Назначение: Асинхронная обработка документов с поддержкой полного NLP-пайплайна

import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from celery import shared_task, Task
from celery.exceptions import MaxRetriesExceededError

# Импорт компонентов обработки
from core.text_processing.chunker import SemanticChunker, ChunkingStrategy
from ingest.text_extraction import extract_text_from_file, FileType
from ingest.embedding import EmbeddingGenerator
from processing.ner import EntityExtractor
from processing.graph_builder import KnowledgeGraphBuilder
from db.operations import DocumentStorage
from utils.monitoring import TaskMonitor
from utils.error_handling import DocumentProcessingError

logger = logging.getLogger(__name__)

class DocumentProcessingTask(Task):
    """Базовый класс для задач обработки документов с расширенной логикой"""
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Обработка неудачного выполнения задачи"""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=True)
        TaskMonitor.track_failure(task_id, str(exc))
        super().on_failure(exc, task_id, args, kwargs, einfo)

@shared_task(bind=True, base=DocumentProcessingTask)
def process_document_async(self, doc_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Полный пайплайн асинхронной обработки документа:
    1. Извлечение текста
    2. Семантическое чанкование
    3. Генерация эмбеддингов
    4. Извлечение сущностей
    5. Построение графа знаний
    6. Сохранение результатов
    
    Args:
        doc_info: Словарь с параметрами документа:
            - doc_path: Путь к файлу
            - original_filename: Исходное имя файла
            - chunk_size: Размер чанков (по умолчанию 512)
            - min_confidence: Минимальная уверенность для сущностей (0.7)
            - filters: Фильтры сущностей
            - strategy: Стратегия чанкования (hybrid)
            - user_id: ID пользователя
            - session_id: ID сессии
    
    Returns:
        Словарь с результатами обработки
    
    Raises:
        DocumentProcessingError: При критических ошибках обработки
    """
    try:
        # Валидация входных данных
        doc_path = Path(doc_info['doc_path'])
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        
        # Инициализация компонентов
        monitor = TaskMonitor(self.request.id)
        chunker = SemanticChunker()
        embedder = EmbeddingGenerator()
        entity_extractor = EntityExtractor()
        graph_builder = KnowledgeGraphBuilder()
        storage = DocumentStorage()
        
        # 1. Отслеживание начала задачи
        monitor.start_processing(
            filename=doc_info.get('original_filename', doc_path.name),
            user_id=doc_info.get('user_id'),
            session_id=doc_info.get('session_id')
        )
        
        # 2. Извлечение текста
        self.update_state(state='PROGRESS', meta={'stage': 'extracting_text'})
        text = extract_text_from_file(
            doc_path,
            file_type=FileType.from_path(doc_path)
        )
        monitor.log_stage_completion('text_extraction')
        
        # 3. Чанкование
        self.update_state(state='PROGRESS', meta={'stage': 'chunking'})
        chunks = chunker.chunk_text(
            text=text,
            strategy=doc_info.get('strategy', 'hybrid'),
            chunk_size=doc_info.get('chunk_size', 512),
            semantic_threshold=0.85,
            min_chunk_size=32
        )
        monitor.log_stage_completion('chunking', metrics={'chunks_count': len(chunks)})
        
        # 4. Генерация эмбеддингов
        self.update_state(state='PROGRESS', meta={'stage': 'embedding'})
        embedder.generate_embeddings(chunks)
        monitor.log_stage_completion('embedding')
        
        # 5. Извлечение сущностей
        self.update_state(state='PROGRESS', meta={'stage': 'entity_extraction'})
        entity_extractor.extract_entities(
            chunks,
            min_confidence=doc_info.get('min_confidence', 0.7),
            entity_filters=doc_info.get('filters', [])
        )
        monitor.log_stage_completion('entity_extraction')
        
        # 6. Построение графа знаний
        self.update_state(state='PROGRESS', meta={'stage': 'graph_building'})
        graph = graph_builder.build_graph(chunks)
        monitor.log_stage_completion('graph_building')
        
        # 7. Сохранение результатов
        self.update_state(state='PROGRESS', meta={'stage': 'saving_results'})
        doc_id = storage.save_document(
            chunks=chunks,
            graph=graph,
            doc_name=doc_info.get('original_filename', doc_path.name),
            metadata={
                'user_id': doc_info.get('user_id'),
                'session_id': doc_info.get('session_id'),
                'processing_time': monitor.get_processing_time(),
                'chunking_strategy': doc_info.get('strategy', 'hybrid')
            }
        )
        monitor.log_stage_completion('saving_results')
        
        # Финализация задачи
        result = {
            'status': 'completed',
            'document_id': doc_id,
            'chunks_count': len(chunks),
            'entities_count': sum(len(chunk.entities) for chunk in chunks),
            'processing_time': monitor.get_processing_time(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        monitor.complete_processing(result)
        return result
        
    except Exception as e:
        error_msg = f"Document processing failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        monitor.log_error(error_msg)
        
        try:
            self.retry(exc=DocumentProcessingError(error_msg), countdown=60, max_retries=3)
        except MaxRetriesExceededError:
            raise DocumentProcessingError(f"Max retries exceeded for document: {doc_path.name}")