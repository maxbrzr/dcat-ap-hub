"""FileCollection keys and lookup."""

import pytest

from dcat_ap_hub.files.files import FileCollection, scan_directory


@pytest.fixture
def collection(tmp_path):
    # Archives often extract several folders with the same file names.
    for folder in ("run1", "run2"):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "falling1.csv").write_text("a\n1\n")
    (tmp_path / "train.csv").write_text("a\n1\n")
    return FileCollection(tmp_path, scan_directory(tmp_path))


def test_same_named_files_in_subfolders_are_all_kept(collection):
    assert collection.keys() == ["run1/falling1.csv", "run2/falling1.csv", "train.csv"]
    assert len(collection) == 3


def test_lookup_by_path_unique_name_or_unique_fragment(collection, tmp_path):
    assert collection["run2/falling1.csv"].path == tmp_path / "run2" / "falling1.csv"
    assert collection["train.csv"].path == tmp_path / "train.csv"
    assert collection["run1"].path == tmp_path / "run1" / "falling1.csv"


def test_ambiguous_or_missing_names_fail_clearly(collection):
    with pytest.raises(KeyError, match="Ambiguous key 'falling1.csv'.*run1/falling1.csv.*run2/falling1.csv"):
        collection["falling1.csv"]
    with pytest.raises(KeyError, match="not found"):
        collection["missing.csv"]
