# 📄 Файл: schemas.py
# 📂 Путь: db/
# 📌 Назначение: Pydantic-схемы для сериализации и валидации моделей базы знаний

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Union, Dict
from uuid import UUID
from datetime import datetime

class DocumentBase(BaseModel):
    title: str
    source_path: str
    source_type: str
    metadata: Optional[Dict] = {}

class DocumentCreate(DocumentBase):
    original_content: str

class DocumentOut(DocumentBase):
    id: UUID
    processing_status: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True

class ChunkBase(BaseModel):
    content: str
    chunk_type: str = 'text'
    embedding_model: Optional[str] = None
    metadata: Optional[Dict] = {}

class ChunkCreate(ChunkBase):
    document_id: UUID
    embedding_vector: Optional[bytes] = None
    semantic_hash: Optional[str] = None

class ChunkOut(ChunkBase):
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        orm_mode = True

class EntityBase(BaseModel):
    original_text: str
    normalized_text: Optional[str]
    label: str
    label_hierarchy: Optional[List[str]] = []
    confidence: float = 0.0
    metadata: Optional[Dict] = {}

    @validator('confidence')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v

class EntityCreate(EntityBase):
    pass

class EntityOut(EntityBase):
    id: UUID
    first_seen: datetime
    last_seen: Optional[datetime] = None

    class Config:
        orm_mode = True

class DocumentWithChunks(DocumentOut):
    chunks: List[ChunkOut] = []

class DocumentWithEntities(DocumentOut):
    entities: List[EntityOut] = []
