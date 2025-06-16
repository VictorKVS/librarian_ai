# 📄 Файл: documents.py
# 📂 Путь: api/
# 📌 Назначение: API для загрузки и обработки документов с расширенной валидацией и мониторингом

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_202_ACCEPTED,
    HTTP_400_BAD_REQUEST,
    HTTP_413_REQUEST_ENTITY_TOO_LARGE,
    HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    HTTP_500_INTERNAL_SERVER_ERROR
)
from typing import List, Optional
import os
import tempfile
import shutil
import logging
from pathlib import Path
from datetime import datetime

from core.tools.async_tasks import process_document_async
from core.config import settings
from utils.file_utils import clean_temp_files, get_file_extension, validate_file_size
from models.schemas import (
    AsyncTaskResponse,
    DocumentUploadResponse,
    ErrorResponse
)
from .dependencies import verify_api_key

router = APIRouter(
    prefix="/api/v1/documents",
    tags=["Documents"],
    dependencies=[Depends(verify_api_key)]
)

logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc"
}

MAX_FILE_SIZE_MB = settings.MAX_FILE_SIZE_MB

@router.post(
    "/async-process",
    response_model=DocumentUploadResponse,
    responses={
        202: {"description": "Document processing started"},
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def upload_and_process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Document file to process"),
    filename: Optional[str] = Form(None, description="Custom filename"),
    chunk_size: int = Form(
        settings.DEFAULT_CHUNK_SIZE,
        ge=100,
        le=5000,
        description="Size of text chunks in characters"
    ),
    min_confidence: float = Form(
        settings.DEFAULT_MIN_CONFIDENCE,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for entity extraction"
    ),
    filters: List[str] = Form(
        [],
        description="List of entity types to filter"
    ),
    strategy: str = Form(
        "hybrid",
        description="Chunking strategy: fixed|sentence|paragraph|semantic|hybrid"
    )
) -> DocumentUploadResponse:
    """
    Загружает документ и запускает асинхронную обработку:
    1. Валидация файла
    2. Сохранение во временное хранилище
    3. Запуск фоновой задачи обработки
    """
    try:
        if file.content_type not in SUPPORTED_MIME_TYPES:
            raise HTTPException(
                status_code=HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type. Supported types: {list(SUPPORTED_MIME_TYPES.keys())}"
            )

        file.file.seek(0, 2)
        file_size = file.file.tell()
        if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {MAX_FILE_SIZE_MB}MB"
            )
        file.file.seek(0)

        file_ext = get_file_extension(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext, dir=settings.TEMP_DIR) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        background_tasks.add_task(
            clean_temp_files,
            temp_path,
            timeout=settings.TEMP_FILE_CLEANUP_TIMEOUT
        )

        task = process_document_async.delay(
            doc_path=temp_path,
            original_filename=filename or file.filename,
            chunk_size=chunk_size,
            min_confidence=min_confidence,
            filters=filters,
            strategy=strategy
        )

        logger.info(
            f"Started processing task {task.id} for file {filename or file.filename}",
            extra={
                "task_id": task.id,
                "filename": filename or file.filename,
                "size": file_size,
                "timestamp": datetime.utcnow().isoformat()
            }
        )

        return DocumentUploadResponse(
            task_id=task.id,
            status_url=f"/api/v1/tasks/{task.id}",
            message="Document processing started successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Document processing failed: {str(e)}",
            exc_info=True,
            extra={
                "filename": filename or file.filename if 'file' in locals() else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during document processing"
        )
