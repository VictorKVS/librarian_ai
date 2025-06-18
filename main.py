import argparse
from loader import load_file
from parser import parse_text
from chunker import split_into_chunks
from extractor import extract_entities
from summary_generator import generate_summary
from graph_tools import build_knowledge_graph


def process_document(path: str):
    print(f"\n📂 Загружается файл: {path}")
    raw_text = load_file(path)

    print("✂️ Очистка и парсинг...")
    parsed_text = parse_text(raw_text)

    print("🔪 Делим на чанки...")
    chunks = split_into_chunks(parsed_text)
    print(f"  -> Чанков: {len(chunks)}")

    print("🧬 Извлечение сущностей...")
    entities = extract_entities(parsed_text)
    print(f"  -> Найдено: {len(entities)} сущностей")

    print("📄 Генерация TL;DR...")
    summary = generate_summary(parsed_text)
    print(f"  -> Сводка: {summary[:200]}...")

    print("📈 Построение графа знаний...")
    graph = build_knowledge_graph(entities)
    print(f"  -> Узлов: {len(graph.nodes())}, Связей: {len(graph.edges())}")

    print("✅ Готово!")
    return {
        "summary": summary,
        "entities": entities,
        "graph": graph
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="📚 Librarian Mini")
    parser.add_argument("path", type=str, help="Путь к файлу для обработки")
    args = parser.parse_args()

    process_document(args.path)
