# -*- coding: utf-8 -*-
# 📄 Файл: embedder.py
# 📂 Путь: core/tools/embedder.py
# 📌 Назначение: Генерация эмбеддингов текста с помощью Sentence-Transformers

from typing import List, Optional, Union
import logging
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError(
        "SentenceTransformer не установлен. Установите пакет:\n"
        "    pip install sentence-transformers"
    )

# Логгер
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class EmbeddingService:
    """
    Сервис генерации эмбеддингов для текста.

    Использует SentenceTransformer (например, 'all-MiniLM-L6-v2').
    Поддерживает одиночные строки и батчи. Есть опция нормализации векторов.

    Пример:
        embedder = EmbeddingService()
        vec = embedder.embed_text("Пример текста")
        batch = embedder.embed_batch(["текст 1", "текст 2"])
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize_embeddings: bool = True,
        **model_kwargs
    ):
        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model_kwargs = model_kwargs

        logger.info(f"Загрузка модели эмбеддинга: {model_name} → {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device, **model_kwargs)
            # Пробный эмбеддинг
            test_vec = self.model.encode("тест", normalize_embeddings=normalize_embeddings)
            self.embedding_dim = test_vec.shape[0]
            logger.info(f"✅ Загружено. Размерность векторов: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке модели: {e}")
            raise

    def embed_text(
        self,
        text: str,
        normalize: Optional[bool] = None,
        convert_to_numpy: bool = True
    ) -> Union[List[float], np.ndarray]:
        """
        Эмбеддинг одной строки текста.

        Args:
            text: Текст для преобразования.
            normalize: Принудительная нормализация (иначе — из init).
            convert_to_numpy: Вернуть np.array или list.

        Returns:
            Вектор эмбеддинга.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Текст должен быть непустой строкой.")

        normalize = self.normalize_embeddings if normalize is None else normalize

        try:
            vec = self.model.encode(
                text,
                normalize_embeddings=normalize,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            return np.array(vec) if convert_to_numpy else vec
        except Exception as e:
            logger.error(f"❌ Ошибка эмбеддинга текста: {e}")
            raise

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: Optional[bool] = None,
        convert_to_numpy: bool = True
    ) -> Union[np.ndarray, List[List[float]]]:
        """
        Эмбеддинг списка текстов.

        Args:
            texts: Список строк.
            batch_size: Размер батча.
            normalize: Принудительная нормализация.
            convert_to_numpy: Вернуть np.array или list of lists.

        Returns:
            Массив эмбеддингов.
        """
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise ValueError("texts должен быть списком строк.")

        normalize = self.normalize_embeddings if normalize is None else normalize

        try:
            result = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            return np.array(result) if convert_to_numpy else result
        except Exception as e:
            logger.error(f"❌ Ошибка батч-эмбеддинга: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """
        Возвращает размерность эмбеддингов.
        """
        return self.embedding_dim

    def __repr__(self):
        return (
            f"<EmbeddingService(model_name={self.model_name}, "
            f"device={self.device}, "
            f"embedding_dim={self.embedding_dim})>"
        )
