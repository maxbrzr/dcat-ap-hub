"""Downloading from a local HTTP server, folder naming, and processing after download."""

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from dcat_ap_hub import Dataset
from dcat_ap_hub.metadata.constants import PROCESSOR_PROFILE_URI
from dcat_ap_hub.metadata.transfer import dataset_dir_name

PROCESSOR = """
from pathlib import Path

def process(input_files, output_dir):
    rows = [p.name.split("_")[0] for p in sorted(input_files)]
    Path(output_dir, "out.txt").write_text("\\n".join(rows))
"""


class _Handler(SimpleHTTPRequestHandler):
    # Serve scripts like raw.githubusercontent.com does.
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".py": "text/plain",
        ".jsonld": "application/ld+json",
        # Hugging Face serves Parquet as a generic binary.
        ".parquet": "application/octet-stream",
    }

    def log_message(self, *args):
        pass


@pytest.fixture
def server(tmp_path):
    root = tmp_path / "remote"
    root.mkdir()
    handler = functools.partial(_Handler, directory=str(root))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def _publish(root: Path, base: str, *, title: str, processor_title: str) -> str:
    (root / "a.csv").write_text("x\n1\n")
    (root / "b.csv").write_text("x\n2\n")
    (root / "proc.py").write_text(PROCESSOR)
    record = {
        "@graph": [
            {
                "@id": f"{base}/dataset",
                "@type": "dcat:Dataset",
                "dct:title": title,
                "dct:relation": {"@id": f"{base}/dataset#processor"},
            },
            {
                "@type": "dcat:Distribution",
                "dct:title": "part.csv",
                "dcat:downloadURL": [
                    {"@id": f"{base}/a.csv"},
                    {"@id": f"{base}/b.csv"},
                ],
            },
            {
                "@id": f"{base}/dataset#processor",
                "@type": "rdfs:Resource",
                "dct:title": processor_title,
                "dct:conformsTo": PROCESSOR_PROFILE_URI,
                "dcat:downloadURL": {"@id": f"{base}/proc.py"},
            },
        ]
    }
    (root / "record.jsonld").write_text(json.dumps(record))
    return f"{base}/record.jsonld"


def test_processor_without_suffix_served_as_text_plain_processes(server, tmp_path):
    root, base = server
    url = _publish(root, base, title="Example", processor_title="Processor: Example")

    ds = Dataset.from_url(url)
    files = ds.download(data_dir=tmp_path / "data", verbose=False)

    # The downloader names the script after its MIME type...
    assert any(f.path.suffix == ".txt" for f in files)
    # ...and processing still finds and runs it.
    ds.process(verbose=False)
    assert (ds.processed_path / "out.txt").read_text() == "part\npart"  # processor not among inputs


def test_list_valued_download_url_downloads_every_file(server, tmp_path):
    root, base = server
    url = _publish(root, base, title="Example", processor_title="proc.py")

    ds = Dataset.from_url(url)
    files = ds.download(data_dir=tmp_path / "data", verbose=False)

    contents = sorted(f.path.read_text() for f in files.filter_by("csv"))
    assert contents == ["x\n1\n", "x\n2\n"]


def test_title_with_path_and_windows_characters_is_one_folder(server, tmp_path):
    root, base = server
    url = _publish(
        root, base, title='TCM: Pump/Motor "v2"?', processor_title="proc.py"
    )

    files = Dataset.from_url(url).download(data_dir=tmp_path / "data", verbose=False)

    assert files.root == tmp_path / "data" / "TCM Pump Motor v2"


def test_existing_folder_named_after_raw_title_is_reused(server, tmp_path):
    root, base = server
    url = _publish(root, base, title="TCM: Pump", processor_title="proc.py")
    legacy = tmp_path / "data" / "TCM: Pump"
    legacy.mkdir(parents=True)

    files = Dataset.from_url(url).download(data_dir=tmp_path / "data", verbose=False)

    assert files.root == legacy


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("AIA17 — Manufacturing Enterprises", "AIA17 — Manufacturing Enterprises"),
        ("Industrial Machine (syn)", "Industrial Machine (syn)"),
        ("a/b\\c", "a b c"),
        ("ends with dot.", "ends with dot"),
        ("  ", "untitled_dataset"),
    ],
)
def test_dataset_dir_name(title, expected):
    assert dataset_dir_name(title) == expected


def test_generic_mime_type_keeps_extension_and_html_is_marked(server, tmp_path):
    root, base = server
    (root / "0000.parquet").write_bytes(b"PAR1")
    (root / "login.html").write_text("<html>sign in</html>")
    record = {
        "@graph": [
            {"@id": f"{base}/d", "@type": "dcat:Dataset", "dct:title": "Example"},
            {
                "@type": "dcat:Distribution",
                "dct:title": "default / train / 0000.parquet",
                "dcat:downloadURL": f"{base}/0000.parquet",
            },
            {
                "@type": "dcat:Distribution",
                # Advertised as CSV, but the server answers with a login page.
                "dct:title": "table.csv",
                "dcat:downloadURL": f"{base}/login.html",
            },
        ]
    }
    (root / "record.jsonld").write_text(json.dumps(record))

    files = Dataset.from_url(f"{base}/record.jsonld").download(
        data_dir=tmp_path / "data", verbose=False
    )

    suffixes = sorted(f.path.suffix for f in files if f.path.name != "dcat-metadata.jsonld")
    assert suffixes == [".html", ".parquet"]


def test_access_url_fallback_warns(server, tmp_path, caplog):
    root, base = server
    (root / "landing.html").write_text("<html>record page</html>")
    (root / "data.csv").write_text("x\n1\n")
    record = {
        "@graph": [
            {"@id": f"{base}/d", "@type": "dcat:Dataset", "dct:title": "Example"},
            {
                "@type": "dcat:Distribution",
                "dct:title": "Zenodo record access",
                "dcat:accessURL": f"{base}/landing.html",
            },
            {
                "@type": "dcat:Distribution",
                "dct:title": "data.csv",
                "dcat:accessURL": f"{base}/landing.html",
                "dcat:downloadURL": f"{base}/data.csv",
            },
        ]
    }
    (root / "record.jsonld").write_text(json.dumps(record))

    with caplog.at_level("WARNING", logger="dcat-ap-hub"):
        Dataset.from_url(f"{base}/record.jsonld").download(
            data_dir=tmp_path / "data", verbose=False
        )

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "'Zenodo record access' has no dcat:downloadURL" in warnings[0]
    assert f"{base}/landing.html" in warnings[0]
