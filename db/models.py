# 📄 Файл: models.py
# 📂 Путь: db/
# 📌 Назначение: Расширенные ORM-модели для системы управления знаниями

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, 
    ForeignKey, DateTime, JSON, Text, LargeBinary,
    UniqueConstraint, Index, event
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates, Session
from sqlalchemy.dialects.postgresql import UUID, ARRAY, TSVECTOR
import uuid
import datetime
from typing import Optional, Dict, List, Any
import logging
from config import settings
import numpy as np

Base = declarative_base()
logger = logging.getLogger(__name__)

def generate_uuid() -> str:
    return str(uuid.uuid4())

class KnowledgeDocument(Base):
    __tablename__ = "knowledge_documents"
    __table_args__ = (
        Index('ix_doc_source_composite', 'source_path', 'source_type'),
        Index('ix_doc_processing_status', 'processing_status'),
        Index('ix_doc_content_fts', 'content_fts', postgresql_using='gin'),
        {'schema': settings.DB_KNOWLEDGE_SCHEMA}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(512), nullable=False, index=True)
    original_content = Column(Text, nullable=False)
    processed_content = Column(Text)
    content_fts = Column(TSVECTOR)
    source_path = Column(String(1024), unique=True, nullable=False)
    source_type = Column(String(32), nullable=False)
    processing_status = Column(String(32), default='pending', nullable=False)
    processing_errors = Column(JSON, default=[])
    processing_version = Column(Integer, default=0, nullable=False)
    metadata = Column(JSON, default={}, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)
    expires_at = Column(DateTime)

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan", order_by="DocumentChunk.created_at")
    entities = relationship("KnowledgeEntity", secondary="document_entities", back_populates="documents")

    @validates('source_type')
    def validate_source_type(self, key: str, source_type: str) -> str:
        allowed_types = ['pdf', 'web', 'txt', 'docx', 'markdown', 'email', 'database']
        if source_type not in allowed_types:
            raise ValueError(f"Invalid source type. Allowed: {allowed_types}")
        return source_type

    def update_processing_status(self, status: str, error: Optional[Dict] = None):
        self.processing_status = status
        if error and status == 'failed':
            if not self.processing_errors:
                self.processing_errors = []
            self.processing_errors.append(error)

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        Index('ix_chunk_document_id', 'document_id'),
        Index('ix_chunk_embedding', 'embedding_vector', postgresql_using='hnsw'),
        Index('ix_chunk_semantic_hash', 'semantic_hash'),
        {'schema': settings.DB_KNOWLEDGE_SCHEMA}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.knowledge_documents.id"), nullable=False)
    content = Column(Text, nullable=False)
    chunk_type = Column(String(32), default='text', nullable=False)
    embedding_vector = Column(LargeBinary)
    embedding_model = Column(String(64))
    semantic_hash = Column(String(64), index=True)
    metadata = Column(JSON, default={}, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)

    document = relationship("KnowledgeDocument", back_populates="chunks")
    entities = relationship("KnowledgeEntity", secondary="chunk_entities", back_populates="chunks")

    def get_embedding(self) -> Optional[np.ndarray]:
        if self.embedding_vector:
            return np.frombuffer(self.embedding_vector, dtype=np.float32)
        return None

    def set_embedding(self, embedding: np.ndarray, model_name: str):
        if not isinstance(embedding, np.ndarray):
            raise ValueError("Embedding must be numpy array")
        self.embedding_vector = embedding.astype(np.float32).tobytes()
        self.embedding_model = model_name

class KnowledgeEntity(Base):
    __tablename__ = "knowledge_entities"
    __table_args__ = (
        Index('ix_entity_normalized_text', 'normalized_text'),
        Index('ix_entity_label', 'label'),
        Index('ix_entity_confidence', 'confidence'),
        UniqueConstraint('normalized_text', 'label', name='uq_entity_normalized'),
        {'schema': settings.DB_KNOWLEDGE_SCHEMA}
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_text = Column(String(512), nullable=False)
    normalized_text = Column(String(512), index=True)
    label = Column(String(64), nullable=False)
    label_hierarchy = Column(ARRAY(String), default=[])
    confidence = Column(Float, default=0.0, nullable=False)
    metadata = Column(JSON, default={}, nullable=False)
    first_seen = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, onupdate=datetime.datetime.utcnow)

    documents = relationship("KnowledgeDocument", secondary="document_entities", back_populates="entities")
    chunks = relationship("DocumentChunk", secondary="chunk_entities", back_populates="entities")
    related_entities = relationship("KnowledgeEntity", secondary="entity_relations", primaryjoin="KnowledgeEntity.id==entity_relations.c.entity_id", secondaryjoin="KnowledgeEntity.id==entity_relations.c.related_id", backref="related_to")

    @validates('confidence')
    def validate_confidence(self, key: str, confidence: float) -> float:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return confidence

class DocumentEntity(Base):
    __tablename__ = "document_entities"
    __table_args__ = {'schema': settings.DB_KNOWLEDGE_SCHEMA}

    document_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.knowledge_documents.id"), primary_key=True)
    entity_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.knowledge_entities.id"), primary_key=True)
    frequency = Column(Integer, default=1, nullable=False)
    contexts = Column(ARRAY(Text), default=[])

class ChunkEntity(Base):
    __tablename__ = "chunk_entities"
    __table_args__ = {'schema': settings.DB_KNOWLEDGE_SCHEMA}

    chunk_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.document_chunks.id"), primary_key=True)
    entity_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.knowledge_entities.id"), primary_key=True)
    count = Column(Integer, default=1)

class EntityRelation(Base):
    __tablename__ = "entity_relations"
    __table_args__ = {'schema': settings.DB_KNOWLEDGE_SCHEMA}

    entity_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.knowledge_entities.id"), primary_key=True)
    related_id = Column(UUID(as_uuid=True), ForeignKey(f"{settings.DB_KNOWLEDGE_SCHEMA}.knowledge_entities.id"), primary_key=True)
    relation_type = Column(String(64), nullable=False)
    confidence = Column(Float, default=0.0)
    evidence = Column(ARRAY(Text), default=[])

@event.listens_for(KnowledgeDocument, 'before_update')
def update_content_fts(mapper, connection, target):
    if target.processed_content:
        stmt = f"""
            UPDATE {settings.DB_KNOWLEDGE_SCHEMA}.knowledge_documents 
            SET content_fts = to_tsvector('english', processed_content)
            WHERE id = :id
        """
        connection.execute(stmt, {'id': target.id})

@event.listens_for(Session, 'after_flush')
def update_entity_last_seen(session, context):
    for instance in session.new:
        if isinstance(instance, (DocumentEntity, ChunkEntity)):
            instance.entity.last_seen = datetime.datetime.utcnow()
