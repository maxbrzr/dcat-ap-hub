"""Processor discovery, metadata parsing, and local processing end to end."""

import json
from pathlib import Path

import pytest

from dcat_ap_hub import Dataset, add_processor
from dcat_ap_hub.metadata.constants import PROCESSOR_PROFILE_URI, SPLIT_FILES
from dcat_ap_hub.metadata.parsing import _parse_json_content
from dcat_ap_hub.processing.processor import apply_processor_logic

# Writes the split files plus a listing of the raw CSVs it saw, recursively.
FUNCTION_PROCESSOR = """
from pathlib import Path

def process(data_dir, output_dir):
    tables = sorted(
        str(p.relative_to(data_dir)) for p in Path(data_dir).rglob("*.csv")
        if Path(output_dir) not in p.parents
    )
    Path(output_dir, "seen.txt").write_text("\\n".join(tables))
    for name in ("train.csv", "test.csv"):
        Path(output_dir, name).write_text("a\\n1\\n")
"""

FUNCTION_AND_HELPER_CLASS = """
from pathlib import Path

class Splitter:  # a helper that happens to have process(); must be ignored
    def process(self, data_dir, output_dir):
        Path(output_dir, "out.txt").write_text("class")

def process(data_dir, output_dir):
    Path(output_dir, "out.txt").write_text("function")
"""

SUBCLASS_PROCESSOR = """
from pathlib import Path
from dcat_ap_hub import DataProcessor

class Splits(DataProcessor):
    def process(self, data_dir, output_dir):
        Path(output_dir, "out.txt").write_text("subclass")
"""

IMPORTED_CLASS_ONLY = """
from pathlib import PurePath  # has no process(); must not be picked up
"""


def _run(tmp_path: Path, source: str) -> str:
    script = tmp_path / "proc.py"
    script.write_text(source)
    out = tmp_path / "out"
    out.mkdir()
    apply_processor_logic(script, tmp_path, out)
    return (out / "out.txt").read_text()


def test_module_level_function_wins_over_helper_classes(tmp_path):
    assert _run(tmp_path, FUNCTION_AND_HELPER_CLASS) == "function"


def test_data_processor_subclass_works(tmp_path):
    assert _run(tmp_path, SUBCLASS_PROCESSOR) == "subclass"


def test_script_without_processor_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="process\\(data_dir, output_dir\\)"):
        _run(tmp_path, IMPORTED_CLASS_ONLY)


def _metadata(processor_entry: dict, *, dataset_extra: dict | None = None) -> dict:
    return {
        "@graph": [
            {
                "@id": "https://example.org/dataset/1",
                "@type": "dcat:Dataset",
                "dct:title": "Example",
                **(dataset_extra or {}),
            },
            {
                "@type": "dcat:Distribution",
                "dct:title": "table.csv",
                "dct:format": "CSV",
                "dcat:downloadURL": "https://example.org/table.csv",
            },
            processor_entry,
        ]
    }


PROCESSOR_ENTRY = {
    "@id": "https://example.org/dataset/1#processor",
    "@type": "rdfs:Resource",
    "dct:title": "processor.py",
    "dct:conformsTo": {"@id": PROCESSOR_PROFILE_URI},
    "dcat:downloadURL": "processor.py",
}


def test_processor_profile_is_hammerhai_vocabulary():
    assert PROCESSOR_PROFILE_URI == "https://data.hammerhai.eu/vocab/Processor"


def test_parses_other_prefixes_and_compact_profile():
    document = {
        "@context": {
            "dcterms": "http://purl.org/dc/terms/",
            "d": "http://www.w3.org/ns/dcat#",
        },
        "@graph": [
            {
                "@id": "https://example.org/dataset/1",
                "@type": "d:Dataset",
                "dcterms:title": "Example",
                "dcterms:relation": {"@id": "https://example.org/dataset/1#processor"},
            },
            {
                "@type": "http://www.w3.org/ns/dcat#Distribution",
                "http://purl.org/dc/terms/title": "table.csv",
                "d:downloadURL": "https://example.org/table.csv",
            },
            {
                "@id": "https://example.org/dataset/1#processor",
                "@type": "http://www.w3.org/2000/01/rdf-schema#Resource",
                "dcterms:title": "processor.py",
                "dcterms:conformsTo": "hammerhai:Processor",
                "d:accessURL": "https://example.org/processor.py",
            },
        ],
    }
    meta = _parse_json_content(document, "test")
    assert meta.title == "Example"
    assert [d.title for d in meta.distributions] == ["table.csv"]
    [resource] = meta.related_resources
    assert resource.role == "processor"
    assert resource.linked
    assert resource.download_url == "https://example.org/processor.py"


def test_resource_without_processor_profile_is_not_a_processor():
    entry = {key: value for key, value in PROCESSOR_ENTRY.items() if key != "dct:conformsTo"}
    [resource] = _parse_json_content(_metadata(entry), "test").related_resources
    assert resource.role == "other"


def test_relative_urls_resolve_against_a_remote_record_only():
    remote = _parse_json_content(_metadata(PROCESSOR_ENTRY), "https://host.example/run/dcat-ap-metadata.jsonld")
    local = _parse_json_content(_metadata(PROCESSOR_ENTRY), "/data/run/dcat-ap-metadata.jsonld")
    assert remote.related_resources[0].download_url == "https://host.example/run/processor.py"
    assert local.related_resources[0].download_url == "processor.py"
    assert remote.related_resources[0].get_filename() == "processor.py"


def test_linked_processor_is_preferred(tmp_path):
    unlinked = {**PROCESSOR_ENTRY, "@id": "x", "dct:title": "other.py", "dcat:downloadURL": "other.py"}
    document = _metadata(
        PROCESSOR_ENTRY,
        dataset_extra={"dct:relation": {"@id": PROCESSOR_ENTRY["@id"]}},
    )
    document["@graph"].insert(1, unlinked)
    (tmp_path / "dcat-ap-metadata.jsonld").write_text(json.dumps(document))
    (tmp_path / "table.csv").write_text("a\n1\n")
    (tmp_path / "processor.py").write_text(FUNCTION_PROCESSOR)
    (tmp_path / "other.py").write_text("raise SystemExit('wrong processor')")

    Dataset.from_directory(tmp_path).process(verbose=False)

    assert (tmp_path / "processed" / "seen.txt").read_text() == "table.csv"


def test_processor_gets_the_folder_including_extracted_subfolders(tmp_path):
    (tmp_path / "dcat-ap-metadata.jsonld").write_text(json.dumps(_metadata(PROCESSOR_ENTRY)))
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "part.csv").write_text("a\n1\n")
    (tmp_path / "processor.py").write_text(FUNCTION_PROCESSOR)

    files = Dataset.from_directory(tmp_path).process(verbose=False)

    assert (tmp_path / "processed" / "seen.txt").read_text() == "archive/part.csv"
    assert {name for name in SPLIT_FILES} <= {f.path.name for f in files}


def test_processor_must_write_the_split_files(tmp_path):
    (tmp_path / "dcat-ap-metadata.jsonld").write_text(json.dumps(_metadata(PROCESSOR_ENTRY)))
    (tmp_path / "processor.py").write_text(
        "from pathlib import Path\n"
        "def process(data_dir, output_dir):\n"
        "    Path(output_dir, 'X_train.npy').write_bytes(b'')\n"
    )

    with pytest.raises(RuntimeError, match="did not write train.csv, test.csv"):
        Dataset.from_directory(tmp_path).process(verbose=False)
    # A partial output must not count as processed on the next call.
    with pytest.raises(RuntimeError, match="did not write"):
        Dataset.from_directory(tmp_path).process(verbose=False)


def test_add_processor_works_with_any_prefixes_and_is_idempotent(tmp_path):
    raw = {
        "@context": {"dcterms": "http://purl.org/dc/terms/", "d": "http://www.w3.org/ns/dcat#"},
        "@graph": [
            {
                "@id": "https://example.org/dataset/1",
                "@type": "http://www.w3.org/ns/dcat#Dataset",
                "dcterms:title": "Example",
                "dcterms:relation": {"@id": "https://example.org/other"},
            }
        ],
    }

    document = add_processor(add_processor(raw))

    datasets = [n for n in document["@graph"] if n.get("@type") == "http://www.w3.org/ns/dcat#Dataset"]
    assert len(datasets) == 1  # found despite full IRIs; no second dataset added
    assert datasets[0]["dcterms:relation"] == [
        {"@id": "https://example.org/other"},
        {"@id": "https://example.org/dataset/1#processor"},
    ]
    assert sum(1 for n in document["@graph"] if n.get("@id", "").endswith("#processor")) == 1
    meta = _parse_json_content(document, "test")
    [resource] = meta.related_resources
    assert (resource.role, resource.linked, resource.download_url) == ("processor", True, "processor.py")


def test_add_processor_creates_a_dataset_for_empty_records():
    document = add_processor({}, dataset_id="https://example.org/d", title="Fallback")
    meta = _parse_json_content(document, "test")
    assert meta.title == "Fallback"
    assert meta.related_resources[0].linked


def test_list_valued_urls_are_not_dropped():
    document = _metadata(
        {**PROCESSOR_ENTRY, "dcat:downloadURL": [{"@id": "https://example.org/p.py"}]}
    )
    document["@graph"][1]["dcat:downloadURL"] = [
        {"@id": "https://example.org/a.csv"},
        "https://example.org/b.csv",
    ]
    meta = _parse_json_content(document, "test")
    assert [d.download_url for d in meta.distributions] == [
        "https://example.org/a.csv",
        "https://example.org/b.csv",
    ]
    assert meta.related_resources[0].download_url == "https://example.org/p.py"


def test_distribution_without_download_url_keeps_access_url():
    document = _metadata(PROCESSOR_ENTRY)
    del document["@graph"][1]["dcat:downloadURL"]
    document["@graph"][1]["dcat:accessURL"] = "https://example.org/landing"
    [distribution] = _parse_json_content(document, "test").distributions
    assert distribution.best_url == "https://example.org/landing"


def test_first_of_several_datasets_is_used():
    document = _metadata(PROCESSOR_ENTRY)
    document["@graph"].append(
        {"@id": "https://example.org/dataset/2", "@type": "dcat:Dataset", "dct:title": "Second"}
    )
    assert _parse_json_content(document, "test").title == "Example"


def test_csv_with_late_non_ascii_bytes_and_parquet_load(tmp_path):
    import pandas as pd

    from dcat_ap_hub.files.files import LazyFile

    csv = tmp_path / "table.csv"
    csv.write_text("a,b\n" + "1,x\n" * 50_000 + "2,café\n", encoding="utf-8")
    parquet = tmp_path / "table.parquet"
    pd.DataFrame({"a": [1, 2]}).to_parquet(parquet)

    assert LazyFile(csv).data.iloc[-1]["b"] == "café"
    assert LazyFile(parquet).data["a"].tolist() == [1, 2]


def test_ai_ready_record_is_its_own_dataset_derived_from_the_raw_one():
    from dcat_ap_hub import ai_ready_record

    raw = {
        "@graph": [
            {
                "@id": "https://piveau.io/set/data/raw",
                "@type": "dcat:Dataset",
                "dct:title": {"@language": "en", "@value": "Pump data"},
                "dcat:distribution": {"@id": "https://piveau.io/set/distribution/1"},
            },
            {
                "@id": "https://piveau.io/set/distribution/1",
                "@type": "dcat:Distribution",
                "dct:title": "table.csv",
                "dcat:downloadURL": {"@id": "https://portal.example/table.csv"},
            },
            {"@id": "https://catalog.example/record", "foaf:primaryTopic": {"@id": "https://piveau.io/set/data/raw"}},
        ]
    }

    document = add_processor(ai_ready_record(raw))
    meta = _parse_json_content(document, "https://host.example/run/dcat-ap-metadata.jsonld")

    assert meta.title == "Pump data (AI-ready)"
    assert meta.derived_from == ["https://piveau.io/set/data/raw"]
    # The raw files are still the processor's inputs.
    assert [d.download_url for d in meta.distributions] == ["https://portal.example/table.csv"]
    [processor] = meta.related_resources
    assert processor.linked and processor.download_url == "https://host.example/run/processor.py"
    ids = [node.get("@id") for node in document["@graph"]]
    assert "#dataset" in ids and "#processor" in ids
    assert "https://piveau.io/set/data/raw" not in ids
    # References to the raw dataset follow the new id; the raw record is untouched.
    assert document["@graph"][2]["foaf:primaryTopic"] == {"@id": "#dataset"}
    assert raw["@graph"][0]["@id"] == "https://piveau.io/set/data/raw"


def test_ai_ready_record_needs_a_dataset():
    from dcat_ap_hub import ai_ready_record

    with pytest.raises(ValueError, match="no dcat:Dataset"):
        ai_ready_record({})
