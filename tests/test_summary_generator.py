"""
🧪 Тесты для модуля генерации суммаризации (summary_generator.py)

Цель:
- Проверить базовую генерацию суммаризации
- Убедиться в корректной структуре результата
- Протестировать поведение с кастомными ролями
"""

import pytest
from core.tools.summary_generator import (
    UniversalSummaryGenerator,
    SummaryRequest,
    SummaryConfig,
    SummaryLength,
    SummaryStyle,
    Role
)

class DummyLLM:
    """🧪 Заглушка LLM-клиента"""
    def generate(self, prompt: str) -> str:
        return "Сгенерированная суммаризация для теста."

@pytest.fixture
def generator():
    gen = UniversalSummaryGenerator()
    gen.llm = DummyLLM()
    return gen

def test_basic_summary_generation(generator):
    """Тест базовой генерации одной суммаризации"""
    request = SummaryRequest(
        text="Это текст для тестирования работы генератора суммаризации. Он содержит достаточно символов, чтобы пройти валидацию.",
        language="ru",
        config=SummaryConfig(length=SummaryLength.SHORT, style=SummaryStyle.SIMPLE),
        roles=[Role.GENERAL]
    )

    result = generator.generate_summary(request)

    assert result.language == "ru"
    assert Role.GENERAL.value in result.summaries
    assert isinstance(result.summaries[Role.GENERAL.value], str)
    assert len(result.key_points) >= 0
    assert result.time_taken >= 0

def test_custom_role_summary(generator):
    """Тест генерации суммаризации с кастомной ролью"""
    request = SummaryRequest(
        text="Некоторый длинный текст для проверки кастомной роли. Проверка генерации текста для разных потребителей.",
        language="en",
        config=SummaryConfig(length=SummaryLength.MEDIUM),
        custom_roles=["UX_researcher"]
    )

    result = generator.generate_summary(request)

    assert "UX_researcher" in result.summaries
    assert isinstance(result.summaries["UX_researcher"], str)
