# chat.deepseek.com
# 📄 Файл: semantic_search.py
# 📂 Путь: core/tools/
# 📌 Назначение: Продвинутый семантический поиск с гибридными стратегиями и кэшированием

from typing import List, Dict, Optional, Union, AsyncGenerator
from enum import Enum
from dataclasses import dataclass
import logging
import numpy as np
from functools import lru_cache
from pydantic import BaseModel, Field, validator
from datetime import datetime
from config import settings
import asyncio

logger = logging.getLogger(__name__)

class SearchStrategy(str, Enum):
    """Стратегии семантического поиска"""
    DENSE = "dense"      # Плотные векторные эмбеддинги
    SPARSE = "sparse"    # Разреженные (TF-IDF/BM25)
    HYBRID = "hybrid"    # Гибридный подход
    RERANK = "rerank"    # С переранжированием

class SearchParams(BaseModel):
    """Параметры поиска с валидацией"""
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=100)
    strategy: SearchStrategy = SearchStrategy.HYBRID
    min_score: float = Field(0.3, ge=0, le=1)
    filters: Optional[Dict[str, Union[str, List[str]]]] = None
    use_cache: bool = Field(True)

    @validator('query')
    def validate_query(cls, v):
        cleaned = v.strip()
        if len(cleaned) < 2:
            raise ValueError("Query too short")
        return cleaned

@dataclass
class SearchResult:
    """Результат поиска с расширенными метаданными"""
    id: str
    score: float
    content: str
    metadata: Dict
    vector: Optional[np.ndarray] = None
    keywords: Optional[List[str]] = None
    timestamp: Optional[datetime] = None

class SemanticSearch:
    """
    Усовершенствованный семантический поиск с:
    - Поддержкой нескольких стратегий
    - Кэшированием запросов
    - Гибкой фильтрацией
    - Асинхронным выполнением
    """

    def __init__(self, vector_store_client, text_processor=None):
        """
        Args:
            vector_store_client: Клиент векторного хранилища
            text_processor: Процессор текста для расширенных функций
        """
        self.client = vector_store_client
        self.text_processor = text_processor
        self._setup_logging()

    def _setup_logging(self):
        """Настройка детального логирования"""
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(settings.LOG_LEVEL)

    @lru_cache(maxsize=1000)
    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Кэшированное получение эмбеддинга запроса"""
        if self.text_processor:
            return self.text_processor.embed_query(query)
        return self.client.get_embedding(query)

    async def query(self, params: SearchParams) -> List[SearchResult]:
        """
        Асинхронный поиск с расширенными параметрами.
        
        Args:
            params: Параметры поиска
            
        Returns:
            Отсортированный список результатов SearchResult
        """
        logger.info(f"Search: '{params.query[:50]}...' (strategy={params.strategy})")
        
        try:
            # Выбор стратегии поиска
            if params.strategy == SearchStrategy.DENSE:
                results = await self._dense_search(params)
            elif params.strategy == SearchStrategy.SPARSE:
                results = await self._sparse_search(params)
            elif params.strategy == SearchStrategy.HYBRID:
                results = await self._hybrid_search(params)
            elif params.strategy == SearchStrategy.RERANK:
                results = await self._rerank_search(params)
            else:
                raise ValueError(f"Unknown strategy: {params.strategy}")

            # Применение фильтров
            if params.filters:
                results = self._apply_filters(results, params.filters)
                
            # Фильтрация по min_score
            results = [r for r in results if r.score >= params.min_score]
            
            logger.info(f"Found {len(results)} results")
            return sorted(results, key=lambda x: x.score, reverse=True)[:params.top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise

    async def _dense_search(self, params: SearchParams) -> List[SearchResult]:
        """Плотный векторный поиск"""
        embedding = self._get_query_embedding(params.query)
        raw_results = await self.client.search_dense(
            embedding=embedding,
            top_k=params.top_k * 3,  # Берем больше для последующей фильтрации
            filters=params.filters
        )
        return self._format_results(raw_results)

    async def _hybrid_search(self, params: SearchParams) -> List[SearchResult]:
        """Гибридный поиск (вектор + ключевые слова)"""
        # Векторный поиск
        vector_results = await self._dense_search(params)
        
        # Извлечение ключевых слов
        keywords = []
        if self.text_processor:
            keywords = self.text_processor.extract_keywords(params.query)
        
        if keywords:
            # Комбинирование score
            scored_results = []
            for result in vector_results:
                keyword_score = self._calc_keyword_score(result, keywords)
                combined_score = (result.score * 0.7) + (keyword_score * 0.3)
                scored_results.append(SearchResult(
                    **{**result.__dict__, 'score': combined_score, 'keywords': keywords}
                ))
            return scored_results
            
        return vector_results

    async def batch_query(self, queries: List[SearchParams]) -> Dict[str, List[SearchResult]]:
        """
        Пакетный поиск с прогресс-баром.
        
        Args:
            queries: Список параметров поиска
            
        Returns:
            Словарь {запрос: результаты}
        """
        from tqdm.asyncio import tqdm_asyncio
        
        results = {}
        tasks = [self.query(params) for params in queries]
        
        try:
            completed = await tqdm_asyncio.gather(
                *tasks,
                desc="Processing queries",
                total=len(queries))
            
            for params, result in zip(queries, completed):
                results[params.query] = result
                
        except Exception as e:
            logger.error(f"Batch query failed: {str(e)}")
            raise
            
        return results

    def _format_results(self, raw_results: List[Dict]) -> List[SearchResult]:
        """Форматирование сырых результатов"""
        return [
            SearchResult(
                id=r['id'],
                score=r['score'],
                content=r['content'],
                metadata=r.get('metadata', {}),
                vector=r.get('vector'),
                timestamp=r.get('timestamp')
            ) for r in raw_results
        ]

    def _calc_keyword_score(self, result: SearchResult, keywords: List[str]) -> float:
        """Вычисление score за ключевые слова"""
        if not keywords:
            return 0.0
            
        content = result.content.lower()
        matches = sum(1 for kw in keywords if kw.lower() in content)
        return min(matches / len(keywords), 1.0)

    def _apply_filters(self, results: List[SearchResult], filters: Dict) -> List[SearchResult]:
        """Применение фильтров к результатам"""
        return [
            r for r in results
            if all(
                r.metadata.get(key) in (values if isinstance(values, list) else [values])
                for key, values in filters.items()
            )
        ]