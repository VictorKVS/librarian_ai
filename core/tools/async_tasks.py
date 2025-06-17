# -*- coding: utf-8 -*-
# 📄 Файл: async_tasks.py
# 📂 Путь: core/tools/async_tasks.py
# 📌 Назначение: Асинхронные Celery-задачи для обработки документов и статуса

from celery import Celery
from celery.result import AsyncResult
import logging
from typing import Dict, Any
from datetime import datetime

# Настройка логгера
logger = logging.getLogger(__name__)

# Настройка Celery (брокер и backend — Redis)
celery_app = Celery(
    "librarian_ai",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=600,  # 10 минут максимум
    worker_prefetch_multiplier=1,
)

class DocumentProcessingError(Exception):
    """Исключение для ошибок при обработке документов."""
    pass

@celery_app.task(bind=True, name="process_document")
def process_document_async(self, doc_id: str) -> Dict[str, Any]:
    """
    Асинхронная задача обработки документа по ID.
    Возвращает словарь с логами и статусами.
    """
    logs = []
    started_at = datetime.utcnow().isoformat()

    try:
        def log(stage: str, progress: float):
            logs.append(stage)
            self.update_state(state='PROGRESS', meta={
                "stage": stage,
                "progress": progress,
                "logs": logs,
                "started_at": started_at
            })
            logger.info(f"[{doc_id}] {stage}")

        # === Пример пайплайна ===
        log("📥 Загрузка документа", 0.1)
        # document = load_document(doc_id)

        log("🔍 Чанкование и анализ", 0.4)
        # chunks = chunker(document)

        log("📐 Векторизация", 0.6)
        # embeddings = embed(chunks)

        log("🧠 Аннотирование / извлечение сущностей", 0.8)
        # entities = extract_entities(embeddings)

        log("💾 Сохранение в БД / хранилище", 0.95)
        # store(doc_id, embeddings, entities)

        finished_at = datetime.utcnow().isoformat()
        return {
            "status": "done",
            "doc_id": doc_id,
            "progress": 1.0,
            "started_at": started_at,
            "finished_at": finished_at,
            "logs": logs,
        }

    except Exception as e:
        error_msg = f"[{doc_id}] ❌ Ошибка обработки: {str(e)}"
        logger.error(error_msg, exc_info=True)
        logs.append(error_msg)
        self.update_state(state='FAILURE', meta={
            "error": error_msg,
            "progress": 0.0,
            "logs": logs,
            "started_at": started_at
        })
        raise DocumentProcessingError(error_msg) from e

def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Проверка статуса задачи по ID.
    """
    result = AsyncResult(task_id, app=celery_app)
    status = {
        "task_id": task_id,
        "status": result.status,
        "result": result.result
    }

    if result.status == "PROGRESS":
        status.update(result.info or {})
    elif result.status == "FAILURE":
        status["error"] = str(result.info)

    return status

def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Принудительное завершение задачи по ID.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        result.revoke(terminate=True)
        logger.warning(f"Задача {task_id} отменена вручную")
        return {"task_id": task_id, "success": True, "message": "Задача отменена"}
    except Exception as e:
        logger.error(f"Ошибка отмены задачи {task_id}: {str(e)}")
        return {"task_id": task_id, "success": False, "message": f"Ошибка отмены: {str(e)}"}
