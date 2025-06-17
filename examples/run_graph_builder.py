from core.processing.ner import EntityProcessor
from core.processing.graph_builder import RelationGraphBuilder

text = "Алексей Навальный и Владимир Путин участвовали в дебатах. Навальный связался с Марией."

ner = EntityProcessor()
entities = ner.extract_entities(text)

graph_builder = RelationGraphBuilder()
G = graph_builder.build_interaction_graph(entities)

# Печать ключевых сущностей по степени центральности
for name, score in graph_builder.get_key_entities(G):
    print(f"{name}: {score:.2f}")
