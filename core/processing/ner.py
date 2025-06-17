# core/processing/ner.py
import spacy
from typing import List, Dict, Set, Optional
from collections import defaultdict
import re
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityProcessor:
    def __init__(self, model_name: str = "ru_core_news_md"):
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Successfully loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Model {model_name} not found. Downloading...")
            from spacy.cli import download
            download(model_name)
            self.nlp = spacy.load(model_name)

        self.patterns = {
            'PHONE': r'\+?[\d\-\(\)\s]{7,15}',
            'CRYPTO_WALLET': r'(bc1|[13])[a-zA-HJ-NP-Z0-9]{25,39}',
            'IBAN': r'[A-Z]{2}\d{2}[A-Z\d]{1,30}',
            'EMAIL': r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+',
            'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'DATE': r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b'
        }

        self.entity_cache = defaultdict(dict)
        self.last_processed = None

    def extract_entities(self, text: str, use_cache: bool = True) -> List[Dict]:
        if use_cache and text in self.entity_cache:
            return self.entity_cache[text]

        start_time = datetime.now()
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
                "source": "spacy"
            })

        for label, pattern in self.patterns.items():
            try:
                for match in re.finditer(pattern, text):
                    entities.append({
                        "text": match.group(),
                        "label": label,
                        "start": match.start(),
                        "end": match.end(),
                        "source": "regex"
                    })
            except re.error as e:
                logger.warning(f"Regex error for pattern {label}: {e}")

        filtered_entities = self._filter_overlapping(entities)
        self.entity_cache[text] = filtered_entities
        self.last_processed = start_time

        logger.info(f"Processed text in {(datetime.now() - start_time).total_seconds():.2f}s")
        return filtered_entities

    def _filter_overlapping(self, entities: List[Dict]) -> List[Dict]:
        entities_sorted = sorted(entities, key=lambda x: (x['end']-x['start'], x['start']), reverse=True)
        filtered = []
        used_positions = set()

        for ent in entities_sorted:
            if not any(pos in used_positions for pos in range(ent['start'], ent['end'])):
                filtered.append(ent)
                used_positions.update(range(ent['start'], ent['end']))

        return sorted(filtered, key=lambda x: x['start'])

    def get_entity_statistics(self) -> Dict:
        stats = defaultdict(int)
        for entities in self.entity_cache.values():
            for ent in entities:
                stats[ent['label']] += 1
        return dict(stats)


# core/processing/graph_builder.py
import networkx as nx
from typing import List, Dict, Tuple
from itertools import combinations
import community as community_louvain
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class RelationGraphBuilder:
    def __init__(self, window_size: int = 5, min_weight: int = 2):
        self.window_size = window_size
        self.min_weight = min_weight
        self.graph_cache = {}

    def build_interaction_graph(self, entities: List[Dict]) -> nx.Graph:
        cache_key = hash(tuple((e['text'], e['label']) for e in entities))
        if cache_key in self.graph_cache:
            return self.graph_cache[cache_key].copy()

        G = nx.Graph()
        entity_texts = [e["text"] for e in entities]

        for ent in entities:
            if not G.has_node(ent["text"]):
                G.add_node(ent["text"],
                           label=ent["label"],
                           count=1,
                           examples=[ent["text"]])
            else:
                G.nodes[ent["text"]]["count"] += 1
                if ent["text"] not in G.nodes[ent["text"]]["examples"]:
                    G.nodes[ent["text"]]["examples"].append(ent["text"])

        co_occurrence = defaultdict(int)
        for i in range(len(entity_texts)):
            for j in range(i+1, min(i+self.window_size, len(entity_texts))):
                pair = tuple(sorted((entity_texts[i], entity_texts[j])))
                co_occurrence[pair] += 1

        for (node1, node2), weight in co_occurrence.items():
            if weight >= self.min_weight and node1 != node2:
                G.add_edge(node1, node2, weight=weight)

        if len(G.nodes) > 0:
            G = self._apply_community_detection(G)
            self._calculate_centrality(G)

        self.graph_cache[cache_key] = G.copy()
        return G

    def _apply_community_detection(self, G: nx.Graph) -> nx.Graph:
        try:
            if len(G.nodes) > 2:
                partition = community_louvain.best_partition(G)
                for node in G.nodes():
                    G.nodes[node]["community"] = partition[node]
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
        return G

    def _calculate_centrality(self, G: nx.Graph):
        try:
            degree_cent = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)

            for node in G.nodes():
                G.nodes[node]["degree_centrality"] = degree_cent.get(node, 0)
                G.nodes[node]["betweenness"] = betweenness.get(node, 0)
                G.nodes[node]["closeness"] = closeness.get(node, 0)
        except Exception as e:
            logger.error(f"Centrality calculation failed: {e}")

    def get_key_entities(self, G: nx.Graph, top_n: int = 5,
                        metric: str = "degree_centrality") -> List[Tuple[str, float]]:
        if metric not in ["degree_centrality", "betweenness", "closeness"]:
            metric = "degree_centrality"

        nodes_with_scores = [(n, G.nodes[n].get(metric, 0)) for n in G.nodes()]
        return sorted(nodes_with_scores, key=lambda x: x[1], reverse=True)[:top_n]

    def visualize_graph(self, G: nx.Graph, output_path: str = None):
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))

            pos = nx.spring_layout(G)
            node_colors = [G.nodes[n].get('community', 0) for n in G.nodes()]

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, cmap=plt.cm.tab20)
            nx.draw_networkx_edges(G, pos, alpha=0.3)
            nx.draw_networkx_labels(G, pos, font_size=8)

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"Graph saved to {output_path}")
            else:
                plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for graph visualization")
        except Exception as e:
            logger.error(f"Graph visualization failed: {e}")
