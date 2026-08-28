"""Detect a corrupted or silently revised download, not just a truncated one.

``ensure_data`` already resumes an interrupted download, and a file count catches
the obvious failure -- 38,000 admissions arriving where 40,336 were listed. It
catches nothing else. A file truncated mid-write by a full disk, a byte flipped in
transit, or an upstream revision that changes values under a stable filename all
produce a complete-looking dataset and a model trained on something other than
what the published numbers describe.

PhysioNet does not publish per-file checksums for this release. Its
``SHA256SUMS.txt`` is three lines, covering ``LICENSE.txt`` and two SVG diagrams --
not one of the 40,336 patient files. So there is nothing upstream to verify
against, and the manifest here is a *self*-check: it pins what this repository
downloaded, so that a later rebuild on the same machine, or a fresh clone on
another, can tell whether it received the same bytes.

The digest is rolled up per hospital rather than stored per file. 40,336 hashes
would be a 2 MB committed artifact whose diff nobody reads; one digest per
hospital either matches or does not, and when it does not, ``verify`` re-walks the
files to name the ones that moved.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import CFG, Config

MANIFEST = "data_checksums.json"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hospital_digest(directory: Path) -> dict:
    """One rolled-up digest over a hospital's admission files.

    The roll-up hashes ``name:digest`` lines in sorted filename order, so it is
    sensitive to a renamed, added or removed file as well as to changed content,
    and it does not depend on the order the filesystem happens to return.
    """
    paths = sorted(directory.glob("p*.psv"))
    if not paths:
        raise FileNotFoundError(f"no .psv files in {directory}")

    rollup = hashlib.sha256()
    total = 0
    for path in paths:
        rollup.update(f"{path.name}:{file_digest(path)}\n".encode())
        total += path.stat().st_size
    return {"sha256": rollup.hexdigest(), "n_files": len(paths), "total_bytes": total}


def manifest_path(cfg: Config = CFG) -> Path:
    # Deliberately not under reports/ or data/: `make clean` empties the first and
    # `.gitignore` excludes the second, and a manifest that either command can
    # remove is not a check on anything.
    return cfg.root / "configs" / MANIFEST


def write_manifest(hospitals=("A", "B"), cfg: Config = CFG, quiet: bool = False) -> Path:
    entries = {h: hospital_digest(cfg.raw_dir / f"training_set{h}") for h in hospitals}
    path = manifest_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    if not quiet:
        for h, e in entries.items():
            print(f"[integrity] hospital {h}: {e['n_files']:,} files, "
                  f"{e['total_bytes'] / 1e6:.1f} MB, sha256 {e['sha256'][:12]}…", flush=True)
    return path


def verify(hospitals=("A", "B"), cfg: Config = CFG, quiet: bool = False) -> dict[str, bool]:
    """Check each hospital against the committed manifest.

    Missing manifest is not a failure: a clone that downloads the data before this
    check existed has nothing to compare against, and refusing to run would make
    the check worse than useless. A *mismatch* is a failure, and it raises.
    """
    path = manifest_path(cfg)
    if not path.exists():
        if not quiet:
            print(f"[integrity] no manifest at {path}; nothing to verify against. "
                  f"Write one with `sepsis data --write-checksums`.", flush=True)
        return {}

    expected = json.loads(path.read_text())
    results, failures = {}, []
    for h in hospitals:
        if h not in expected:
            continue
        directory = cfg.raw_dir / f"training_set{h}"
        if not directory.exists():
            continue
        actual = hospital_digest(directory)
        ok = actual["sha256"] == expected[h]["sha256"]
        results[h] = ok
        if not quiet:
            print(f"[integrity] hospital {h}: {'ok' if ok else 'MISMATCH'} "
                  f"({actual['n_files']:,} files)", flush=True)
        if not ok:
            failures.append(_describe(h, expected[h], actual, directory))

    if failures:
        raise ValueError(
            "downloaded data does not match the committed manifest:\n"
            + "\n".join(failures)
            + "\n\nEvery published number in this repository describes the data in "
              "the manifest. Re-download, or update the manifest deliberately if "
              "the source genuinely changed."
        )
    return results


def _describe(hospital: str, expected: dict, actual: dict, directory: Path) -> str:
    """Say what moved. A bare digest mismatch tells nobody where to look."""
    if actual["n_files"] != expected["n_files"]:
        return (
            f"  hospital {hospital}: {actual['n_files']:,} files present, "
            f"{expected['n_files']:,} expected — the download is incomplete or has "
            f"gained files"
        )
    delta = actual["total_bytes"] - expected["total_bytes"]
    if delta:
        return (
            f"  hospital {hospital}: same {actual['n_files']:,} files, but "
            f"{delta:+,} bytes — contents changed, not the file list"
        )
    return (
        f"  hospital {hospital}: same file count and same total size, but a "
        f"different digest — bytes moved within the files ({directory})"
    )
