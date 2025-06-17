"""
tests/test_chunker.py
🧪 Тесты для модуля чанкинга текста (chunker.py)

Цель:
- Проверить корректность разбиения длинного текста на чанки
- Убедиться, что чанки не превышают заданный лимит символов
"""

import pytest
from core.text_processing.chunker import chunk_text

@pytest.mark.parametrize("input_text, max_len, expected_chunks", [
    (
        "Это предложение номер один. А это второе. И третье здесь. Далее четвёртое. Пятое завершает мысль.",
        50,
        2  # По 2–3 предложения на чанк
    ),
    (
        "Слишком короткий текст.",
        100,
        1  # Один короткий чанк
    ),
    (
        "Первый абзац.\n\nВторой абзац, который намного длиннее и содержит больше предложений для анализа. " * 5,
        200,
        3  # Ожидаем более 1 чанка
    )
])
def test_chunk_text(input_text, max_len, expected_chunks):
    """
    Проверяет, что функция чанкинга:
    - Делит текст на чанки
    - Уважает лимит длины
    - Возвращает ожидаемое количество чанков (приближенно)
    """
    chunks = chunk_text(input_text, max_length=max_len)

    assert isinstance(chunks, list), "Результат должен быть списком"
    assert all(isinstance(c, str) for c in chunks), "Каждый чанк должен быть строкой"
    assert all(len(c) <= max_len for c in chunks), "Ни один чанк не должен превышать max_length"
    assert len(chunks) == expected_chunks or abs(len(chunks) - expected_chunks) <= 1, "Количество чанков отличается от ожидаемого"
