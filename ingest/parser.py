# 📄 Файл: parser.py
# 📂 Путь: ingest/
# 📌 Назначение: Комплексная очистка и нормализация текста с поддержкой мультиязычности. Удаляет HTML-сущности, email, URL, спецсимволы, повторяющуюся пунктуацию, фильтрует короткие и дублирующие абзацы. Готовит текст для чанкования и извлечения сущностей.

import re
import unicodedata
import html
from typing import Optional, List
from functools import lru_cache

class TextParser:
    WHITESPACE_PATTERN = re.compile(r'\s+')
    CONTROL_CHARS_PATTERN = re.compile(r'[\x00-\x1f\x7f-\x9f]')
    UNWANTED_SYMBOLS_PATTERN = re.compile(r'[^\w\s-.,!?;:\'"«»“”()\[\]{}]')
    EMAIL_PATTERN = re.compile(r'\S+@\S+')
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    PHONE_PATTERN = re.compile(r'[+(]?[0-9][0-9-()]{8,}[0-9]')
    REPEAT_PUNCT_PATTERN = re.compile(r'([.,!?;:])\1+')

    def __init__(self, replace_emails=True, replace_urls=True, replace_phones=True, min_paragraph_length=50):
        self.replace_emails = replace_emails
        self.replace_urls = replace_urls
        self.replace_phones = replace_phones
        self.min_paragraph_length = min_paragraph_length

    @lru_cache(maxsize=1000)
    def parse_text(self, text: str, language: Optional[str] = None) -> str:
        if not text or not isinstance(text, str):
            return ""

        text = html.unescape(text)
        text = unicodedata.normalize('NFKC', text)
        text = self.CONTROL_CHARS_PATTERN.sub(' ', text)

        if self.replace_emails:
            text = self.EMAIL_PATTERN.sub('[EMAIL]', text)
        if self.replace_urls:
            text = self.URL_PATTERN.sub('[URL]', text)
        if self.replace_phones:
            text = self.PHONE_PATTERN.sub('[PHONE]', text)

        text = self._clean_unwanted_symbols(text, language)
        paragraphs = self._process_paragraphs(text)
        return self._finalize_text(paragraphs)

    def _clean_unwanted_symbols(self, text: str, language: Optional[str]) -> str:
        text = self.UNWANTED_SYMBOLS_PATTERN.sub(' ', text)
        if language == 'ru':
            text = re.sub(r'[^а-яёА-ЯЁ0-9\s\-.,!?;:\'"«»“”()\[\]{}]', ' ', text)
        elif language == 'en':
            text = re.sub(r'[^a-zA-Z0-9\s\-.,!?;:\'"“”()\[\]{}]', ' ', text)
        return text

    def _process_paragraphs(self, text: str) -> List[str]:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip() and len(p.strip()) >= self.min_paragraph_length]
        unique_paragraphs = []
        seen = set()
        for p in paragraphs:
            simple_p = self.WHITESPACE_PATTERN.sub('', p).lower()
            if simple_p not in seen:
                seen.add(simple_p)
                unique_paragraphs.append(p)
        return unique_paragraphs

    def _finalize_text(self, paragraphs: List[str]) -> str:
        text = '\n'.join(paragraphs)
        text = self.WHITESPACE_PATTERN.sub(' ', text)
        text = self.REPEAT_PUNCT_PATTERN.sub(r'\1', text)
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
        return text.strip()

    def batch_parse(self, texts: List[str], language: Optional[str] = None) -> List[str]:
        return [self.parse_text(text, language) for text in texts]
    

    
    

