# 📄 Файл: storage.py
# 📂 Путь: db/storage.py
# 📌 Назначение: Унифицированный интерфейс для работы с векторными хранилищами

from typing import List, Dict, Optional, Union, Tuple, Generator
import numpy as np
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# Импорт бэкендов
from storage.faiss_index import FaissVectorStore
from storage.pgvector_store import PgVectorStore
from config import settings

logger = logging.getLogger(__name__)

class StorageBackendType(Enum):
    """Типы поддерживаемых хранилищ векторов"""
    FAISS = auto()
    PGVECTOR = auto()
    HYBRID = auto()
    MEMORY = auto()

@dataclass
class SearchResult:
    """Результат векторного поиска с метаданными и статистикой"""
    id: Union[str, int]
    vector: np.ndarray
    metadata: Dict
    score: float
    backend: str
    distance_metric: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'score': self.score,
            'metadata': self.metadata,
            'backend': self.backend,
            'distance_metric': self.distance_metric
        }

class BaseVectorStore(ABC):
    @abstractmethod
    def add(self, vectors: List[np.ndarray], metadata: List[Dict]) -> List[Union[str, int]]:
        pass

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 5, distance_metric: str = 'cosine', **kwargs) -> List[SearchResult]:
        pass

    @abstractmethod
    def delete(self, ids: List[Union[str, int]]) -> int:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> Dict:
        pass

    @abstractmethod
    def batch_search(self, queries: List[np.ndarray], top_k: int = 5, **kwargs) -> List[List[SearchResult]]:
        pass

class HybridVectorStore(BaseVectorStore):
    def __init__(self, faiss_config: Dict = None, pg_config: Dict = None):
        self.faiss = FaissVectorStore(**(faiss_config or {}))
        self.pgvector = PgVectorStore(**(pg_config or {}))
        self.logger = logging.getLogger(f"{__name__}.HybridVectorStore")
        self._executor = ThreadPoolExecutor(max_workers=4)

    def add(self, vectors: List[np.ndarray], metadata: List[Dict]) -> List[Tuple[str, str]]:
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have same length")
        faiss_future = self._executor.submit(self.faiss.add, vectors, metadata)
        pg_future = self._executor.submit(self.pgvector.add, vectors, metadata)
        faiss_ids = faiss_future.result()
        pg_ids = pg_future.result()
        if len(faiss_ids) != len(pg_ids):
            self.logger.warning(f"ID count mismatch: FAISS={len(faiss_ids)}, pgvector={len(pg_ids)}")
        return list(zip(faiss_ids, pg_ids))

    def search(self, query: np.ndarray, top_k: int = 5, distance_metric: str = 'cosine', strategy: str = 'union', **kwargs) -> List[SearchResult]:
        strategies = {
            'union': self._search_union,
            'faiss_first': partial(self._search_priority, primary='faiss'),
            'pg_first': partial(self._search_priority, primary='pgvector'),
            'weighted': self._search_weighted
        }
        search_fn = strategies.get(strategy, self._search_union)
        return search_fn(query, top_k, distance_metric, **kwargs)

    def _search_union(self, query: np.ndarray, top_k: int, distance_metric: str, **kwargs) -> List[SearchResult]:
        faiss_future = self._executor.submit(self.faiss.search, query, top_k, distance_metric, **kwargs)
        pg_future = self._executor.submit(self.pgvector.search, query, top_k, distance_metric, **kwargs)
        results = faiss_future.result() + pg_future.result()
        seen_ids = set()
        unique_results = []
        for res in sorted(results, key=lambda x: x.score, reverse=True):
            if res.id not in seen_ids:
                seen_ids.add(res.id)
                unique_results.append(res)
                if len(unique_results) >= top_k:
                    break
        return unique_results

    def batch_search(self, queries: List[np.ndarray], top_k: int = 5, **kwargs) -> List[List[SearchResult]]:
        half = len(queries) // 2
        faiss_queries = queries[:half]
        pg_queries = queries[half:]
        faiss_future = self._executor.submit(self.faiss.batch_search, faiss_queries, top_k, **kwargs)
        pg_future = self._executor.submit(self.pgvector.batch_search, pg_queries, top_k, **kwargs)
        return faiss_future.result() + pg_future.result()

    def delete(self, ids: List[Tuple[str, str]]) -> int:
        faiss_ids, pg_ids = zip(*ids)
        faiss_future = self._executor.submit(self.faiss.delete, faiss_ids)
        pg_future = self._executor.submit(self.pgvector.delete, pg_ids)
        return min(faiss_future.result(), pg_future.result())

    def clear(self) -> None:
        self.faiss.clear()
        self.pgvector.clear()

    def get_stats(self) -> Dict:
        with self._executor:
            faiss_future = self._executor.submit(self.faiss.get_stats)
            pg_future = self._executor.submit(self.pgvector.get_stats)
            return {
                'faiss': faiss_future.result(),
                'pgvector': pg_future.result()
            }

class VectorStorage:
    def __init__(self, backend_type: Union[StorageBackendType, str] = None, config: Optional[Dict] = None):
        config = config or {}
        if backend_type is None:
            backend_type = settings.VECTOR_STORE_BACKEND
        if isinstance(backend_type, str):
            backend_type = StorageBackendType[backend_type.upper()]
        self.backend_type = backend_type
        self.config = config
        if backend_type == StorageBackendType.FAISS:
            self.backend = FaissVectorStore(**config.get('faiss', {}))
        elif backend_type == StorageBackendType.PGVECTOR:
            self.backend = PgVectorStore(**config.get('pgvector', {}))
        elif backend_type == StorageBackendType.HYBRID:
            self.backend = HybridVectorStore(faiss_config=config.get('faiss', {}), pg_config=config.get('pgvector', {}))
        else:
            raise ValueError(f"Unsupported backend: {backend_type}")
        self.logger = logging.getLogger(f"{__name__}.VectorStorage")
        self.logger.info(f"Initialized {backend_type.name} storage")

    def add_vectors(self, vectors: List[np.ndarray], metadata: List[Dict], batch_size: int = 1000) -> List[Union[str, int, Tuple]]:
        if len(vectors) != len(metadata):
            raise ValueError("Vectors and metadata must have same length")
        results = []
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_metadata = metadata[i:i+batch_size]
            try:
                results.extend(self.backend.add(batch_vectors, batch_metadata))
            except Exception as e:
                self.logger.error(f"Failed to add batch {i//batch_size}: {str(e)}")
                raise
        return results

    def search_similar(self, query: np.ndarray, top_k: int = 5, distance_metric: str = 'cosine', **kwargs) -> List[SearchResult]:
        try:
            results = self.backend.search(query, top_k=top_k, distance_metric=distance_metric, **kwargs)
            self.logger.debug(f"Found {len(results)} similar vectors")
            return results
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}", exc_info=True)
            raise

    def batch_search(self, queries: List[np.ndarray], top_k: int = 5, **kwargs) -> List[List[SearchResult]]:
        return self.backend.batch_search(queries, top_k, **kwargs)

    def migrate_to_backend(self, new_backend: Union[StorageBackendType, str], new_config: Optional[Dict] = None) -> None:
        if isinstance(new_backend, str):
            new_backend = StorageBackendType[new_backend.upper()]
        if new_backend == self.backend_type:
            self.logger.warning("Migration to same backend skipped")
            return
        self.logger.info(f"Starting migration to {new_backend.name}")
        if not hasattr(self.backend, 'export_all'):
            raise NotImplementedError("Current backend doesn't support export")
        vectors, metadata = self.backend.export_all()
        new_storage = VectorStorage(new_backend, new_config or self.config)
        try:
            import tqdm
            with tqdm.tqdm(total=len(vectors), desc="Migrating vectors") as pbar:
                for i in range(0, len(vectors), 1000):
                    batch = vectors[i:i+1000]
                    meta_batch = metadata[i:i+1000]
                    new_storage.add_vectors(batch, meta_batch)
                    pbar.update(len(batch))
        except ImportError:
            new_storage.add_vectors(vectors, metadata)
        self.backend = new_storage.backend
        self.backend_type = new_backend
        self.logger.info("Migration completed successfully")

if __name__ == "__main__":
    config = {
        'faiss': {'index_path': '/data/faiss_index'},
        'pgvector': {'dbname': 'vectors', 'user': 'vector_user'}
    }
    storage = VectorStorage('hybrid', config)
    vectors = [np.random.rand(768).astype(np.float32) for _ in range(100)]
    metadata = [{'doc_id': f'doc_{i}', 'content': f'Content {i}'} for i in range(100)]
    storage.add_vectors(vectors, metadata)
    query = np.random.rand(768).astype(np.float32)
    results = storage.search_similar(query, top_k=3)
    for res in results:
        print(f"ID: {res.id}, Score: {res.score:.3f}, Backend: {res.backend}")
