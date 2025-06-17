"""
tests/test_loader.py
🧪 Тесты для модуля загрузки текста (loader.py)

Цель:
- Проверить корректность извлечения текста из различных типов файлов
- Убедиться, что возвращается непустой результат
- Убедиться, что загрузчик обрабатывает разные категории текста (технический, гуманитарный и др.)

Тестовые файлы находятся в папке: test_data/
"""

import pytest
from pathlib import Path
from core.tools.loader import load_text_from_file

# 📂 Путь к директории с тестовыми данными
TEST_DIR = Path(__file__).parent / "test_data"

# 🧪 Параметризованные тесты: (название файла, ожидаемая подстрока в тексте)
@pytest.mark.parametrize(
    "filename,expected_phrase",
    [
        ("computers.txt", "процессор"),       # технический текст
        ("humans.txt", "эмоции"),             # гуманитарный
        ("empty.txt", ""),                    # пустой файл (граничный случай)
        ("symbols_only.txt", ""),             # файл без текста
        ("multilang.txt", "интеллект")        # мультилингвальный текст
    ]
)
def test_load_text_from_file(filename: str, expected_phrase: str):
    """
    Проверяет, что загрузчик:
    - Возвращает строку
    - Содержит ожидаемый фрагмент (если он задан)
    """
    file_path = TEST_DIR / filename
    text = load_text_from_file(file_path)

    assert isinstance(text, str), "Результат должен быть строкой"
    assert len(text) > 0 or expected_phrase == "", "Текст не должен быть пустым, если файл не пустой"

    if expected_phrase:
        assert expected_phrase in text.lower(), f"Ожидаемая фраза '{expected_phrase}' не найдена"
