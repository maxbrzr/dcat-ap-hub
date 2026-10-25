"""Processor discovery, metadata parsing, and local processing end to end."""

import json
from pathlib import Path

import pytest

from dcat_ap_hub import Dataset
from dcat_ap_hub.metadata.constants import PROCESSOR_PROFILE_URI
from dcat_ap_hub.metadata.parsing import _parse_json_content
from dcat_ap_hub.processing.processor import apply_processor_logic

FUNCTION_PROCESSOR = """
from pathlib import Path

def process(input_files, output_dir):
    rows = [p.name for p in sorted(input_files)]
    Path(output_dir, "out.txt").write_text("\\n".join(rows))
"""

DUCK_CLASS_PROCESSOR = """
from pathlib import Path

class Processor:
    def process(self, input_files, output_dir):
        Path(output_dir, "out.txt").write_text("class")

def process(input_files, output_dir):
    Path(output_dir, "out.txt").write_text("function")
"""

SUBCLASS_PROCESSOR = """
from pathlib import Path
from dcat_ap_hub import DataProcessor

class Legacy(DataProcessor):
    def process(self, input_files, output_dir):
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
    apply_processor_logic(script, [tmp_path / "b.csv", tmp_path / "a.csv"], out)
    return (out / "out.txt").read_text()


def test_module_level_function_needs_no_dcat_ap_hub_import(tmp_path):
    assert _run(tmp_path, FUNCTION_PROCESSOR) == "a.csv\nb.csv"


def test_class_with_process_method_wins_over_function(tmp_path):
    assert _run(tmp_path, DUCK_CLASS_PROCESSOR) == "class"


def test_legacy_data_processor_subclass_still_works(tmp_path):
    assert _run(tmp_path, SUBCLASS_PROCESSOR) == "subclass"


def test_script_without_processor_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="process\\(input_files, output_dir\\)"):
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
    "dct:title": "parser.py",
    "dct:conformsTo": PROCESSOR_PROFILE_URI,
    "dcat:downloadURL": "parser.py",
}


def test_parses_other_prefixes_and_full_iris():
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
                "dcterms:title": "parser.py",
                "d:accessURL": "https://example.org/parser.py",
            },
        ],
    }
    meta = _parse_json_content(document, "test")
    assert meta.title == "Example"
    assert [d.title for d in meta.distributions] == ["table.csv"]
    [resource] = meta.related_resources
    assert resource.role == "processor"
    assert resource.linked
    assert resource.download_url == "https://example.org/parser.py"


def test_linked_processor_is_preferred(tmp_path):
    unlinked = {**PROCESSOR_ENTRY, "@id": "x", "dct:title": "other.py", "dcat:downloadURL": "other.py"}
    document = _metadata(
        PROCESSOR_ENTRY,
        dataset_extra={"dct:relation": {"@id": PROCESSOR_ENTRY["@id"]}},
    )
    document["@graph"].insert(1, unlinked)
    (tmp_path / "dcat-metadata.jsonld").write_text(json.dumps(document))
    (tmp_path / "table.csv").write_text("a\n1\n")
    (tmp_path / "parser.py").write_text(FUNCTION_PROCESSOR)
    (tmp_path / "other.py").write_text("raise SystemExit('wrong processor')")

    Dataset.from_directory(tmp_path).process(verbose=False)

    assert (tmp_path / "processed" / "out.txt").read_text() == "other.py\ntable.csv"


def test_local_folder_with_plain_processor_name_processes(tmp_path):
    (tmp_path / "dcat-metadata.jsonld").write_text(json.dumps(_metadata(PROCESSOR_ENTRY)))
    (tmp_path / "unrelated.jsonld").write_text("{}")
    (tmp_path / "table.csv").write_text("a\n1\n")
    (tmp_path / "parser.py").write_text(FUNCTION_PROCESSOR)

    Dataset.from_directory(tmp_path).process(verbose=False)

    # Processor and metadata sidecar are excluded from the inputs.
    assert (tmp_path / "processed" / "out.txt").read_text() == "table.csv\nunrelated.jsonld"


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
