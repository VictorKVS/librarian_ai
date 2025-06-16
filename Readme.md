🧠 Librarian AI — Минимальная RAG-платформа нового поколения

Librarian AI — интеллектуальный агент, который превращает документы (PDF, DOCX, TXT, HTML и др.) в структурированную базу знаний с сущностями, цитатами, определениями и графами смыслов.

🚀 Что делает агент:

Загружает документы любой длины

Разбивает их на смысловые чанки

Извлекает ключевые сущности, определения и цитаты

Преобразует текст в векторные представления (RAG)

Создаёт граф знаний со связями между понятиями

Сохраняет всё в базу данных (SQLite/PostgreSQL + FAISS/pgvector)

Доступен через CLI/API/Telegram

🧱 Архитектура:

Модульная структура: ingest/, processing/, db/, api/

Гибкий LLMRouter: поддержка OpenAI, YandexGPT, Local ChatGLM

Основан на SQLAlchemy, FAISS, sentence-transformers

# 📁 Librarian AI — структура проекта (Windows)

librarian_ai/
├── README.md                    # 📘 Главная документация проекта, цели, установка, примеры использования
├── alembic.ini                  # ⚙️ Конфигурация Alembic (путь к env.py, URL базы данных)
│
├── db/                         # 🗃️ Работа с базой данных и моделями
│   ├── __init__.py              # Инициализация пакета db
│   ├── create_tables.py         # Скрипт для создания всех таблиц из моделей вручную
│   ├── models.py                # SQLAlchemy ORM-модели: документы, чанки, сущности и связи
│   ├── schemas.py               # Pydantic-схемы для валидации данных (Document, Chunk, Entity, Graph)
│   ├── session.py               # Асинхронное подключение к БД через SQLAlchemy
│   ├── storage.py               # Унифицированный интерфейс к FAISS, pgvector и гибридному хранилищу
│   └── alembic/                # Alembic: управление миграциями
│       ├── env.py               # Скрипт Alembic, связывает metadata и URL
│       ├── script.py.mako       # Шаблон генерации миграций
│       └── versions/            # Каталог автосгенерированных миграций
│
├── ingest/                     # 📥 Модули загрузки и первичной обработки
│   ├── loader.py               # Загрузка документов из файлов, папок, URL, API
│   ├── parser.py               # Универсальный парсер текстов (PDF, DOCX, HTML, Markdown и т.п.)
│   ├── chunker.py              # Деление текста на семантические чанки
│   ├── async_tasks.py          # Фоновые задачи обработки (Celery)
│   └── ingest_and_index.py     # Скрипт запуска всего пайплайна загрузки и индексации
│
├── processing/                 # 🧠 Модули смысловой обработки
│   └── [будет добавлено]       # Извлечение сущностей (NER), построение графа знаний, связи
│
├── storage/                    # 🔎 Слои векторных хранилищ
│   ├── faiss_index.py          # FAISS backend: хранение и поиск векторов локально
│   ├── pgvector_store.py       # PostgreSQL + pgvector backend: хранение в БД
│   └── [будет добавлено]       # Расширения, кэширование, облачные backend-и
│
├── scripts/                    # 🛠️ Скрипты администрирования и интеграции
│   └── [будет добавлено]       # Очистка, миграции, тестовые загрузки
│
├── cli/                        # 🖥️ CLI-интерфейс
│   └── [будет добавлено]       # Поддержка команд: ingest, reset, migrate, search
│
├── api/                        # 🌐 API-интерфейсы
│   └── [будет добавлено]       # FastAPI endpoints, Telegram API, REST для бэкенда
│
├── tests/                      # ✅ Модульные и интеграционные тесты
│   └── [будет добавлено]       # Pytest тесты для моделей, парсинга, хранилищ и API
│
├── utils/                      # 🔧 Вспомогательные модули и конфигурация
│   ├── config.py               # Pydantic-настройки проекта: пути, API ключи, базы, переменные
│   └── logging.py              # Инициализация и настройка логгера (формат, уровни, вывод)

# Дополнительно (опционально):
# ├── requirements.txt          # Зависимости проекта
# ├── .env                      # Переменные окружения (не добавлять в git!)
# └── Dockerfile / docker-compose.yml  # Контейнеризация и деплой


🎓 Цель

Проект создаётся как учебно-прикладная платформа, с пошаговой архитектурной логикой и возможностью расширения до продакшена (GraphRAG, кластеризация, UI).

Ветка main — стабильная мини-версия. Ветка dev — эксперименты и расширения.

📦 Установка

pip install -r requirements.txt
python db/create_tables.py
python scripts/ingest_and_index.py path/to/document.pdf

📌 Лицензия

MIT — свободно используйте, модифицируйте, развивайте!

## 🧠 Технологии для создания баз знаний

### 1.1. Системы управления базами знаний (KM Systems)

**Готовые платформы:**
- Confluence (Atlassian) – для корпоративных БЗ
- Notion – гибкая платформа для личных и командных БЗ
- Helpjuice, Zendesk Guide – для клиентских баз знаний
- MediaWiki, DokuWiki – открытые wiki-решения
- Obsidian, Logseq – для связанных заметок и графов знаний

**Специализированные KM-решения:**
- ProProfs Knowledge Base
- Bloomfire
- Guru

### 1.2. Искусственный интеллект и NLP

**Обработка естественного языка (NLP):**
- BERT, GPT-4, Claude – автоматический анализ и генерация
- Rasa, Dialogflow – чат-боты в БЗ

**Семантические технологии:**
- OWL, RDF, SPARQL – онтологии и связанные данные
- Neo4j – графовые базы данных

### 1.3. Хранение и структурирование данных

**Базы данных:**
- SQL (PostgreSQL, MySQL) – структурированные данные
- NoSQL (MongoDB, Elasticsearch) – неструктурированные данные

**Графовые БД:**
- Neo4j, Amazon Neptune – сложные взаимосвязи

### 1.4. Инструменты для визуализации знаний
- MindMeister, XMind – ментальные карты
- Miro, Lucidchart – диаграммы
- Obsidian, Roam Research – графы знаний

## 📚 Книги по созданию и управлению базами знаний

### 2.1. Основы управления знаниями (Knowledge Management, KM)
- "Working Knowledge" – Davenport & Prusak
- "The Knowledge Management Toolkit" – Amrit Tiwana
- "Building a Knowledge-Driven Organization" – Robert H. Buckman

### 2.2. Онтологии и семантические технологии
- "Semantic Web for the Working Ontologist" – Allemang, Hendler
- "Ontology Engineering" – Asunción Gómez-Pérez

### 2.3. Искусственный интеллект и NLP
- "Natural Language Processing in Action" – Hobson Lane
- "Speech and Language Processing" – Jurafsky & Martin

### 2.4. UX и дизайн баз знаний
- "Don’t Make Me Think" – Steve Krug
- "Every Page is Page One" – Mark Baker

## 🧩 Методологии и подходы
- Методология CommonKADS – проектирование БЗ
- SCRUM, Agile – гибкая разработка
- Дизайн-мышление – удобные интерфейсы

Да, учитывая дополнения по технологиям и лучшим практикам управления знаниями, в проект Librarian AI можно внести следующие улучшения:

🔧 Технические изменения:
1. Стандартизация под промышленные подходы
Внедрить плагинную архитектуру для parser.py, entity_extractor.py, позволяющую легко добавлять новые типы сущностей.

Добавить поддержку онто-сущностей и связей на базе RDF/OWL (например, через RDFLib или интеграцию с Neo4j).

Поддержка множественных источников данных (например, импорт из Notion, Obsidian, MediaWiki).

2. Расширение API и CLI
CLI: команды для анализа, обновления, синхронизации знаний

FastAPI: эндпоинты для управления сущностями, запросов к графу знаний, генерации ответов на основе reasoning

3. Визуализация
Встроенные экспорты в формат .graphml, .gexf, .json для визуализации графов в Gephi, Obsidian, Neo4j Browser

Генерация SVG/PNG схем через networkx + matplotlib или pyvis

🧠 Методологические изменения:
1. Проектная документация
Добавить файл docs/methodology.md с описанием CommonKADS + KM roadmap

Использовать подход «Every Page is Page One» для микродокументации

2. Управление знаниями
Вести knowledge_index.json — индекс всех сущностей, документов, отношений

Встроить quality score для чанкованного контента: полнота, новизна, связность

3. ML/NLP Roadmap
Добавить слой inference/ для reasoning и semantification

Поддержка стилей вывода: список фактов, аргументация, резюме, гипотезы