# 📦 Установка Librarian AI: Mini Core

## 🧩 Предварительные требования

**Базовые:**
- Python 3.10–3.11 (не поддерживает 3.12)
- Git 2.30+
- SQLite3 (встроен) или PostgreSQL 14–16

**Расширенные (опционально):**
- Redis 6.2+ (для очередей)
- Docker 24.0+ и Docker Compose 2.20+
- FAISS 1.7.4 (для векторного поиска)
- CUDA 11.8 (для GPU)

⚠️ Проверьте версии:
```bash
python --version
docker --version
```

---

## ⚙️ 1. Получение кода
```bash
git clone --branch stable https://github.com/your-org/librarian_ai.git
cd librarian_ai
git submodule update --init  # если есть подмодули
```

---

## 🐍 2. Установка зависимостей

**Вариант A: Виртуальное окружение**
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt --no-cache-dir
```

**Вариант B: Poetry**
```bash
curl -sSL https://install.python-poetry.org | python -
poetry config virtualenvs.in-project true
poetry install --only main
```

---

## 🗃️ 3. Конфигурация
Скопируйте и настройте `.env`:
```bash
cp .env.example .env
nano .env  # или ваш редактор
```

**Критичные настройки:**
```ini
# ДБ (обязательно)
DB_URL=postgresql+asyncpg://user:password@localhost:5432/librarian

# Безопасность
SECRET_KEY=сгенерируйте_через_openssl_rand_hex_32
JWT_ALGORITHM=HS256

# Векторизация
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## 🛠️ 4. Инициализация БД

**Автоматическая настройка:**
```bash
python -m db.setup --init --sample-data
```

**Вручную:**
```bash
alembic upgrade head
python -m db.seed_data  # тестовые данные
```

---

## 🚀 5. Запуск системы

**Стандартный режим:**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 \
  --workers 2 \
  --ssl-keyfile=./certs/key.pem \
  --ssl-certfile=./certs/cert.pem
```

**Фоновые задачи:**
```bash
celery -A core.tasks worker --concurrency=4 -E -P threads
celery -A core.tasks beat --loglevel=info
```

---

## 🐳 Docker-развертывание

**Подготовка:**
```bash
mkdir -p ./data/{db,redis,documents}
```

**Запуск:**
```bash
docker-compose up -d --scale api=2 --scale worker=3
```

**Мониторинг:**
```bash
docker-compose logs -f --tail=50
```

---

## ✅ Верификация установки

**Проверка API:**
```bash
curl -X GET "https://localhost:8000/health" \
  -H "Authorization: Bearer TEST_TOKEN"
```

**Тест обработки:**
```bash
python -m scripts.process_document \
  --file ./examples/sample.pdf \
  --strategy fast
```

**Проверка поиска:**
```bash
python -m scripts.search "AI технологии" --top-k 5
```

---

## 🛠️ Утилиты обслуживания

| Команда | Назначение |
|--------|------------|
| `python -m db.optimize` | Оптимизация БД |
| `python -m scripts.backup --output ./backups` | Резервное копирование |
| `python -m monitoring.disk_usage --threshold 90` | Контроль места |

---

## 🆘 Диагностика проблем

**Сбор логов:**
```bash
python -m diagnostics.collect_logs --output debug.zip
```

**Проверка зависимостей:**
```bash
python -m pip check
```

**Тест производительности:**
```bash
python -m tests.benchmark --workers 10 --requests 100
```

---

## 📌 Важные заметки

Для продакшн-развертывания:
- Используйте reverse proxy (Nginx)
- Настройте автообновление сертификатов
- Регулярно обновляйте зависимости

При обновлениях:
```bash
git pull origin stable
alembic upgrade head
python -m pip install --upgrade -r requirements.txt
```

Контакты поддержки:
- Чат: [Telegram Group]
- Экстренные случаи: admin@librarian-ai.example.com

Полная документация доступна в `docs/advanced_setup.md`