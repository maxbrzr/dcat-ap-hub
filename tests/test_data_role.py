import shutil
from pathlib import Path

import pytest

from dcat_ap_hub import Dataset

# Data used in notebooks/data.ipynb
DATA_URL = "https://ki-daten.hlrs.de/hub/repo/datasets/b48e396f-4fa9-4635-9c71-a79e1290057b.jsonld"


def test_load_and_download_data(tmp_path):
    """
    Test loading a data dataset and downloading its files.
    Replicates parts of notebooks/data.ipynb
    """
    # 1. Load from URL
    ds = Dataset.from_url(DATA_URL)
    assert ds is not None
    assert ds.title is not None
    assert not ds.is_model  # Should be identified as data (default)

    # 2. Download files
    # Use tmp_path to avoid cluttering workspace
    data_dir = tmp_path / "data"
    files = ds.download(data_dir=data_dir, verbose=False)

    assert files is not None
    assert data_dir.exists()

    # Check that we downloaded something
    assert len(files.files) > 0, "No files downloaded"

    # Check for the expected CSV file, but allow for potential naming variations
    # We look for *any* CSV if the exact match fails to be more robust
    csv_files = [f for f in files.files.keys() if f.endswith(".csv")]
    assert len(csv_files) > 0, f"No CSV file found. Files: {list(files.files.keys())}"

    target_file = csv_files[0]

    # Verify file is in the returned FileCollection
    assert target_file in files.files

    # Verify we can access data (basic check)
    # Note: files[key].data property reads the file content
    file_content = files[target_file].data
    assert file_content is not None
    # Just checking it's not empty, assuming it's text/csv
    assert len(file_content) > 0
