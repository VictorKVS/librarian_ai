# 📄 Файл: create_tables.py
# 📂 Путь: db/
# 📌 Назначение: Автоматически создаёт базу данных (SQLite) и таблицы по ORM-моделям из models.py. Получает модели из db.models, создает файл storage/librarian.db.

from sqlalchemy import create_engine
from db.models import Base
import os

# Убедимся, что директория storage/ существует
os.makedirs("storage", exist_ok=True)

# Используем SQLite по умолчанию
DB_URL = "sqlite:///storage/librarian.db"
engine = create_engine(DB_URL)

if __name__ == '__main__':
    print("📦 Создание базы данных и таблиц...")
    Base.metadata.create_all(engine)
    print("✅ База данных успешно инициализирована!")