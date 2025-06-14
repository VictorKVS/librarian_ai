📘 Документ: proekt_librarian_ai.md

🧠 Назначение

Файл proekt_librarian_ai.md служит живым журналом проекта Librarian AI, фиксируя каждое важное решение, реализацию, архитектурный комментарий, а также обучающие пояснения и ссылки на связанные модули. Это — центральная документация, объединяющая цели, ход и результат проекта.

📌 Исходные цели

Librarian AI — агент, который:

принимает документы любой длины;

разбивает их на логические чанки;

извлекает сущности, цитаты, определения;

векторизует данные и строит граф знаний;

сохраняет всё в базу данных и предоставляет доступ через CLI или API.

Проект ориентирован на аналитиков, исследователей, юристов и инженеров, обрабатывающих большие объёмы неструктурированного текста.

🧱 Подтверждённая структура проекта (Librarian AI v1.0)

librarian_ai/
├── db/                        # ORM-модели, миграции, создание базы
├── ingest/                    # Загрузка, парсинг, чанкование, эмбеддинг
├── processing/                # Извлечение знаний, графы
├── storage/                   # Физическое хранилище данных (sqlite, faiss)
├── scripts/                   # Автоматизация пайплайнов
├── cli/                       # Командный интерфейс
├── api/                       # API (FastAPI / Telegram)
├── tests/                     # Тесты
├── utils/                     # Логирование, конфиги
└── README.md                  # Документация

📆 Хронология выполнения

🟢 Июнь 2025

11 июня:

🧱 Подтверждена архитектура проекта от имени "Архитектора";

📁 Структура каталогов зафиксирована;

✅ models.py (ORM) полностью реализован;

🔜 Решение: создавать create_tables.py и ingest_and_index.py;

📘 Добавлен план по ролям: Архитектор, Разработчик, Продуктолог;

💬 Принято создать proekt_librarian_ai.md — живую историю проекта;

📌 Принято описание агента и назначение фиксировать в README.md и e_full.yaml.

🧭 Следующие шаги

✳️ db/create_tables.py — автоматизация инициализации базы;

✳️ scripts/ingest_and_index.py — пайплайн загрузки, обработки и сохранения;

🧪 tests/test_ingest.py — базовые тесты для ingest и embedder;

📘 Обогащение каждого ключевого файла пояснениями (в виде .explain.md).

🔗 Git/GitHub

Репозиторий: github.com/yourusername/librarian_ai.01.mini

Рекомендовано вести коммиты по паттерну:
feat: add embedder, fix: db schema, docs: explain parser

Уровень ветвления: main (рабочий), dev (разработка)

🧩 Примечания

В проекте допускаются экспериментальные файлы в dev/

Все изменения фиксируются в этом файле с аннотацией

Ссылки между модулями указываются через @module.py в комментариях

