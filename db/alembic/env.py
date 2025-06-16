# 📄 Файл: env.py
# 📂 Путь: db/alembic/
# 📌 Назначение: Связывает Alembic с SQLAlchemy metadata и URL базы данных

import logging
from logging.config import fileConfig
from sqlalchemy import engine_from_config, pool
from alembic import context

import sys
import os

# Добавляем корень проекта в PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Инициализация логгера до загрузки настроек
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alembic.env')

try:
    from db.models import Base
    from utils.config import settings
except ImportError as e:
    logger.error("Ошибка импорта модулей приложения: %s", e)
    raise

# Alembic Config object
config = context.config

# Интерпретация файла конфигурации для логирования
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Метаданные для автоматической генерации миграций
target_metadata = Base.metadata

# Установка URL базы данных из настроек
db_url = getattr(settings, 'DATABASE_URL', None)
if not db_url:
    raise ValueError("DATABASE_URL должен быть указан в конфигурации проекта")
config.set_main_option("sqlalchemy.url", db_url)

def include_object(object, name, type_, reflected, compare_to):
    """Фильтрация объектов для включения в миграции."""
    if type_ == "table" and object.schema and object.schema != "public":
        return False
    return True

def run_migrations_offline():
    """Запуск миграций в offline-режиме (без подключения к БД)."""
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=include_object,
        render_as_batch=True  # Поддержка SQLite
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Запуск миграций в online-режиме (с подключением к БД)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            include_object=include_object,
            render_as_batch=True
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    logger.info("Выполнение миграций в offline-режиме")
    run_migrations_offline()
else:
    logger.info("Выполнение миграций в online-режиме")
    run_migrations_online()
