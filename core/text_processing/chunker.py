# 📄 Файл: chunker.py
# 📂 Путь: core/text_processing/
# 📌 Назначение: Интеллектуальное разбиение текста на семантические чанки для обработки в NLP-пайплайне

from typing import List, Dict, Optional, Union, Tuple
import re
from dataclasses import dataclass
from enum import Enum, auto
import spacy
from itertools import zip_longest
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class ChunkingStrategy(Enum):
    """Стратегии семантического чанкования текста"""
    FIXED = auto()           # Фиксированные по размеру чанки
    SENTENCE = auto()        # По границам предложений
    PARAGRAPH = auto()       # По границам параграфов
    SEMANTIC = auto()        # По семантическим кластерам
    HYBRID = auto()          # Комбинированная стратегия

@dataclass
class TextChunk:
    """Семантический чанк текста с расширенными метаданными"""
    id: int
    text: str
    start_pos: int
    end_pos: int
    embeddings: Optional[np.ndarray] = None
    entities: Optional[List[Dict]] = None
    metadata: Dict = None
    cluster_id: Optional[int] = None

class SemanticChunker:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация чанкера с моделями для семантического анализа
        
        Args:
            model_name: Название модели для эмбеддингов
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.nlp = spacy.load('ru_core_news_md')  # Для русского языка
        
    def chunk_text(
        self,
        text: str,
        strategy: Union[ChunkingStrategy, str] = ChunkingStrategy.HYBRID,
        chunk_size: int = 512,
        overlap: int = 64,
        min_chunk_size: int = 32,
        semantic_threshold: float = 0.85,
        **kwargs
    ) -> List[TextChunk]:
        """
        Основной метод для разбиения текста на чанки
        
        Args:
            text: Исходный текст
            strategy: Стратегия чанкования
            chunk_size: Максимальный размер чанка (в токенах)
            overlap: Перекрытие между чанками
            min_chunk_size: Минимальный допустимый размер
            semantic_threshold: Порог семантической схожести
            
        Returns:
            Список текстовых чанков с метаданными
        """
        if isinstance(strategy, str):
            strategy = ChunkingStrategy[strategy.upper()]
            
        # Предварительная обработка текста
        clean_text = self._preprocess_text(text)
        
        if strategy == ChunkingStrategy.FIXED:
            return self._fixed_size_chunking(clean_text, chunk_size, overlap, min_chunk_size)
        elif strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_based_chunking(clean_text, chunk_size, min_chunk_size)
        elif strategy == ChunkingStrategy.SEMANTIC:
            return self._semantic_chunking(clean_text, semantic_threshold, min_chunk_size)
        elif strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunking(clean_text, chunk_size, overlap, semantic_threshold)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

    def _preprocess_text(self, text: str) -> str:
        """Очистка и нормализация текста"""
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _fixed_size_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        min_size: int
    ) -> List[TextChunk]:
        """Фиксированное разбиение по размеру"""
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            
            if len(chunk_words) >= min_size:
                chunk_text = ' '.join(chunk_words)
                chunks.append(TextChunk(
                    id=len(chunks),
                    text=chunk_text,
                    start_pos=start,
                    end_pos=end,
                    metadata={"strategy": "fixed"}
                ))
            start += chunk_size - overlap
            
        return chunks

    def _sentence_based_chunking(
        self,
        text: str,
        max_size: int,
        min_size: int
    ) -> List[TextChunk]:
        """Разбиение по предложениям с учетом максимального размера"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_length = len(sent.split())
            
            if current_length + sent_length > max_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                if len(chunk_text.split()) >= min_size:
                    chunks.append(TextChunk(
                        id=len(chunks),
                        text=chunk_text,
                        start_pos=text.find(current_chunk[0]),
                        end_pos=text.find(current_chunk[-1]) + len(current_chunk[-1]),
                        metadata={"strategy": "sentence"}
                    ))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(sent)
            current_length += sent_length
            
        # Добавляем последний чанк
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= min_size:
                chunks.append(TextChunk(
                    id=len(chunks),
                    text=chunk_text,
                    start_pos=text.find(current_chunk[0]),
                    end_pos=text.find(current_chunk[-1]) + len(current_chunk[-1]),
                    metadata={"strategy": "sentence"}
                ))
                
        return chunks

    def _semantic_chunking(
        self,
        text: str,
        threshold: float,
        min_size: int
    ) -> List[TextChunk]:
        """Семантическое разбиение на основе эмбеддингов"""
        sentences = [sent.text for sent in self.nlp(text).sents]
        if not sentences:
            return []
            
        # Получаем эмбеддинги для предложений
        embeddings = self.embedding_model.encode(sentences)
        
        # Кластеризация K-Means
        optimal_clusters = self._find_optimal_clusters(embeddings)
        kmeans = KMeans(n_clusters=optimal_clusters).fit(embeddings)
        labels = kmeans.labels_
        
        # Группируем предложения по кластерам
        chunks = []
        for cluster_id in set(labels):
            cluster_sents = [sent for sent, label in zip(sentences, labels) if label == cluster_id]
            chunk_text = ' '.join(cluster_sents)
            
            if len(chunk_text.split()) >= min_size:
                start_pos = text.find(cluster_sents[0])
                end_pos = text.find(cluster_sents[-1]) + len(cluster_sents[-1])
                
                chunks.append(TextChunk(
                    id=len(chunks),
                    text=chunk_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    embeddings=self.embedding_model.encode(chunk_text),
                    cluster_id=cluster_id,
                    metadata={"strategy": "semantic"}
                ))
                
        return chunks

    def _hybrid_chunking(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
        threshold: float
    ) -> List[TextChunk]:
        """Гибридная стратегия: семантическая + фиксированная"""
        semantic_chunks = self._semantic_chunking(text, threshold, min_size=10)
        final_chunks = []
        
        for chunk in semantic_chunks:
            if len(chunk.text.split()) <= chunk_size * 1.5:
                final_chunks.append(chunk)
            else:
                # Дополнительно разбиваем большие семантические чанки
                fixed_chunks = self._fixed_size_chunking(
                    chunk.text,
                    chunk_size,
                    overlap,
                    min_size=32
                )
                for fc in fixed_chunks:
                    fc.metadata["parent_cluster"] = chunk.cluster_id
                final_chunks.extend(fixed_chunks)
                
        return final_chunks

    def _find_optimal_clusters(self, embeddings: np.ndarray, max_clusters: int = 10) -> int:
        """Определение оптимального числа кластеров методом локтя"""
        distortions = []
        for k in range(1, min(max_clusters, len(embeddings))):
            kmeans = KMeans(n_clusters=k).fit(embeddings)
            distortions.append(kmeans.inertia_)
            
        # Упрощенный метод нахождения "локтя"
        if len(distortions) >= 3:
            deltas = np.diff(distortions)
            optimal = np.argmin(deltas) + 1
            return max(2, min(optimal, max_clusters))
        return min(2, len(embeddings))

# Пример использования
if __name__ == "__main__":
    chunker = SemanticChunker()
    
    sample_text = """
    Librarian AI - это интеллектуальная система для работы с документами. 
    Она использует современные методы NLP для анализа текстов. 
    Основные функции включают семантический поиск, извлечение сущностей и построение графа знаний.
    
    Система поддерживает несколько стратегий обработки текста. 
    Фиксированное разбиение подходит для технических документов. 
    Семантическое чанкование лучше работает с естественными текстами.
    """
    
    print("=== Fixed Chunking ===")
    fixed_chunks = chunker.chunk_text(sample_text, strategy="FIXED")
    for chunk in fixed_chunks:
        print(f"Chunk {chunk.id}: {chunk.text[:60]}...")
    
    print("\n=== Semantic Chunking ===")
    semantic_chunks = chunker.chunk_text(sample_text, strategy="SEMANTIC")
    for chunk in semantic_chunks:
        print(f"Chunk {chunk.id} (Cluster {chunk.cluster_id}): {chunk.text[:60]}...")