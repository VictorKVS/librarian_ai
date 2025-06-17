# -*- coding: utf-8 -*-
# 📄 Файл: extractor.py
# 📂 Путь: core/tools/extractor.py
# 📌 Назначение: Извлечение сущностей (NER) из текста — пока заглушка

from typing import Dict

def extract_entities(text: str) -> Dict[str, list]:
    """
    📌 Временно заглушка.
    Извлекает именованные сущности из текста (NER).

    В будущем: будет использовать модель SpaCy, HuggingFace или OpenAI для определения сущностей:
        - PERSON, ORG, LOCATION, EVENT, CONCEPT, ...

    Args:
        text (str): Исходный текст

    Returns:
        dict: Категории сущностей и соответствующие элементы
    """
    # TODO: интеграция NER-модели (например, spacy.load("en_core_web_sm") или transformers)
    return {
        "PERSON": [],
        "ORG": [],
        "LOCATION": [],
        "EVENT": [],
        "CONCEPT": [],
    }
