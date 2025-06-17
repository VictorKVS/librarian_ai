import json
import csv
from pathlib import Path

def test_load_index_json():
    path = Path("tests/test_data/osint/index.json")
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)
    assert "osint" in data
    assert len(data["osint"]) >= 5
    for item in data["osint"]:
        assert "filename" in item
        assert "name" in item

def test_load_metadata_csv():
    path = Path("tests/test_data/osint/metadata.csv")
    assert path.exists()
    with open(path, encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
    assert len(reader) >= 5
    required_fields = {"filename", "name", "role", "tags", "source"}
    for row in reader:
        assert required_fields.issubset(row.keys())