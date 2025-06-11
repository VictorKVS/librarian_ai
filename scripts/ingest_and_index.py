# 📄 Файл: ingest_and_index.py
# 📂 Путь: scripts/
# 📌 Назначение: Полный пайплайн обработки документа — загрузка, парсинг, чанкование, эмбеддинг, извлечение сущностей и сохранение в базу. Получает путь к файлу, использует модули из ingest/, db/, processing/.

import sys
from ingest.loader import load_file
from ingest.parser import parse_text
from ingest.chunker import chunk_text
from ingest.embedder import embed_chunks
from processing.entity_extractor import extract_entities
from db.models import KnowledgeDoc, MemoryItem, EntityRecord
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
import uuid
import os

# Настройка подключения к базе
DB_URL = "sqlite:///storage/librarian.db"
engine = create_engine(DB_URL)
Session = sessionmaker(bind=engine)


def process_document(filepath):
    print(f"📥 Обработка файла: {filepath}")
    raw_text, metadata = load_file(filepath)
    parsed_text = parse_text(raw_text)
    chunks = chunk_text(parsed_text)
    embeddings = embed_chunks(chunks)
    entities = extract_entities(parsed_text)

    session = Session()
    doc_id = uuid.uuid4()

    # Создаём запись документа
    doc = KnowledgeDoc(
        id=doc_id,
        title=os.path.basename(filepath),
        content=parsed_text,
        source_path=filepath,
        source_type=metadata.get("type", "txt"),
        processed=True,
        metadata=metadata
    )
    session.add(doc)

    # Добавляем чанки
    for chunk, vector in zip(chunks, embeddings):
        item = MemoryItem(
            content=chunk,
            embedding=vector,
            doc_id=doc_id
        )
        session.add(item)

    # Добавляем сущности
    for ent in entities:
        record = EntityRecord(
            label=ent["label"],
            text=ent["text"],
            confidence=ent.get("confidence", 0.9),
            session_id=None,  # необязательное поле
            metadata=ent.get("meta", {}),
            context=ent.get("context")
        )
        session.add(record)

    session.commit()
    session.close()
    print("✅ Документ успешно обработан и сохранён в базу")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("❗ Использование: python scripts/ingest_and_index.py path/to/file")
        sys.exit(1)

    filepath = sys.argv[1]
    process_document(filepath)
