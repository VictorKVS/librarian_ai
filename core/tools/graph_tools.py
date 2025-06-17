# -*- coding: utf-8 -*-
# 📄 Файл: graph_tools.py
# 📂 Путь: core/tools/graph_tools.py
# 📌 Назначение: Хранение и работа с графом знаний (узлы и связи)

from typing import List, Dict


class GraphStore:
    """
    📘 Класс для построения простого графа знаний:
    - Хранит узлы и связи
    - Позволяет добавлять, экспортировать и фильтровать графовые данные
    """

    def __init__(self):
        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []

    def add_node(self, node_id: str, metadata: Dict) -> None:
        """
        Добавить узел (сущность) в граф.

        Args:
            node_id (str): Уникальный идентификатор узла
            metadata (dict): Дополнительные сведения (например, имя, тип, описание)
        """
        if not node_id or not isinstance(metadata, dict):
            raise ValueError("node_id должен быть строкой, metadata — словарём")
        self.nodes.append({"id": node_id, "meta": metadata})

    def add_edge(self, src: str, dst: str, label: str) -> None:
        """
        Добавить связь между двумя узлами.

        Args:
            src (str): ID исходного узла
            dst (str): ID целевого узла
            label (str): Тип или метка связи
        """
        if not src or not dst or not label:
            raise ValueError("src, dst и label обязательны")
        self.edges.append({"src": src, "dst": dst, "label": label})

    def get_graph(self) -> Dict[str, List[Dict]]:
        """
        Возвращает граф целиком (в виде словаря).

        Returns:
            dict: {'nodes': [...], 'edges': [...]}
        """
        return {
            "nodes": self.nodes,
            "edges": self.edges
        }

    def clear(self) -> None:
        """Очистить граф полностью."""
        self.nodes.clear()
        self.edges.clear()

    def __repr__(self) -> str:
        return f"<GraphStore nodes={len(self.nodes)}, edges={len(self.edges)}>"
