from core.processing.ner import EntityProcessor

text = "Владимир Путин посетил Москву 12 июня 2024 года. Телефон: +7 (495) 123-45-67. Email: example@test.ru"

ner = EntityProcessor()
entities = ner.extract_entities(text)

for ent in entities:
    print(f"{ent['label']:>12}: {ent['text']}")