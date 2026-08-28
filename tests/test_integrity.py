"""Data integrity: catch a download that is complete and wrong.

A file count catches a truncated download. These assertions cover what it does
not: a byte that changed, a file that was renamed, and a manifest that is quietly
absent.
"""

from __future__ import annotations

import json

import pytest

from sepsis.config import Config
from sepsis.data.integrity import hospital_digest, manifest_path, verify, write_manifest


def _hospital(tmp_path, hospital: str = "A", contents=("a", "b", "c")) -> Config:
    cfg = Config(root=tmp_path)
    cfg.ensure_dirs()
    directory = cfg.raw_dir / f"training_set{hospital}"
    directory.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(contents):
        (directory / f"p{i:05d}.psv").write_text(text)
    return cfg


def test_digest_is_stable_across_calls(tmp_path):
    cfg = _hospital(tmp_path)
    directory = cfg.raw_dir / "training_setA"
    assert hospital_digest(directory) == hospital_digest(directory)


def test_digest_changes_when_one_byte_changes(tmp_path):
    cfg = _hospital(tmp_path)
    directory = cfg.raw_dir / "training_setA"
    before = hospital_digest(directory)
    (directory / "p00001.psv").write_text("B")  # same length, different content
    after = hospital_digest(directory)

    assert after["sha256"] != before["sha256"]
    assert after["total_bytes"] == before["total_bytes"], "size alone would not notice"


def test_digest_changes_when_a_file_is_renamed(tmp_path):
    """Same bytes under a different patient id is a different dataset."""
    cfg = _hospital(tmp_path)
    directory = cfg.raw_dir / "training_setA"
    before = hospital_digest(directory)
    (directory / "p00002.psv").rename(directory / "p09999.psv")

    assert hospital_digest(directory)["sha256"] != before["sha256"]


def test_digest_refuses_an_empty_directory(tmp_path):
    cfg = Config(root=tmp_path)
    (cfg.raw_dir / "training_setA").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no .psv files"):
        hospital_digest(cfg.raw_dir / "training_setA")


def test_verify_accepts_the_data_it_pinned(tmp_path):
    cfg = _hospital(tmp_path)
    write_manifest(("A",), cfg, quiet=True)
    assert verify(("A",), cfg, quiet=True) == {"A": True}


def test_verify_rejects_changed_contents_and_names_the_size_shift(tmp_path):
    cfg = _hospital(tmp_path)
    write_manifest(("A",), cfg, quiet=True)
    (cfg.raw_dir / "training_setA" / "p00000.psv").write_text("much longer contents")

    with pytest.raises(ValueError, match="contents changed"):
        verify(("A",), cfg, quiet=True)


def test_verify_rejects_a_missing_file_as_an_incomplete_download(tmp_path):
    cfg = _hospital(tmp_path)
    write_manifest(("A",), cfg, quiet=True)
    (cfg.raw_dir / "training_setA" / "p00000.psv").unlink()

    with pytest.raises(ValueError, match="incomplete"):
        verify(("A",), cfg, quiet=True)


def test_verify_without_a_manifest_is_not_a_failure(tmp_path):
    """A clone whose data predates the manifest has nothing to compare against;
    refusing to run would make the check worse than useless."""
    cfg = _hospital(tmp_path)
    assert verify(("A",), cfg, quiet=True) == {}


def test_manifest_lives_where_neither_clean_nor_gitignore_can_reach_it(tmp_path):
    cfg = Config(root=tmp_path)
    path = manifest_path(cfg)
    assert path.parent.name == "configs"
    assert "reports" not in path.parts and "data" not in path.parts


def test_manifest_records_file_count_and_size_for_a_readable_diff(tmp_path):
    cfg = _hospital(tmp_path, contents=("a", "bb"))
    entry = json.loads(write_manifest(("A",), cfg, quiet=True).read_text())["A"]
    assert entry["n_files"] == 2 and entry["total_bytes"] == 3
