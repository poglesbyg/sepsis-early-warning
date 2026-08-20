"""Fetch the PhysioNet/CinC 2019 training data and cache it as Parquet.

PhysioNet publishes this challenge as ~40,000 individual pipe-separated files --
one per ICU admission -- with no bulk archive. We enumerate them from the
public S3 mirror, pull them concurrently, and collapse each hospital system into
a single tidy Parquet table. Both steps are resumable: an interrupted run picks
up where it left off instead of starting over.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ..config import CFG, Config

S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"
SETS = ("A", "B")


def _get(url: str, retries: int = 4, timeout: int = 60) -> bytes:
    """GET with linear backoff. S3 occasionally throttles a 32-way fan-out."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover
            last = exc
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def list_keys(hospital: str, cfg: Config = CFG) -> list[str]:
    """Page through the S3 bucket listing for one hospital system."""
    prefix = f"{cfg.s3_prefix}/training_set{hospital}/"
    keys: list[str] = []
    token: str | None = None
    while True:
        url = f"{cfg.s3_bucket_url}/?list-type=2&prefix={prefix}&max-keys=1000"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token, safe='')}"
        root = ET.fromstring(_get(url))
        keys.extend(
            el.text for el in root.iter(f"{S3_NS}Key") if el.text and el.text.endswith(".psv")
        )
        truncated = root.findtext(f"{S3_NS}IsTruncated") == "true"
        token = root.findtext(f"{S3_NS}NextContinuationToken")
        if not (truncated and token):
            break
    return sorted(keys)


def _fetch_one(key: str, dest_dir: Path, base_url: str) -> tuple[str, bool]:
    """Return (patient_id, downloaded_now). Existing non-empty files are skipped."""
    name = key.rsplit("/", 1)[-1]
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        return name[:-4], False
    payload = _get(f"{base_url}/{key}")
    tmp = dest.with_suffix(".psv.part")
    tmp.write_bytes(payload)
    tmp.replace(dest)  # atomic: a killed run never leaves a truncated .psv
    return name[:-4], True


def download_hospital(hospital: str, cfg: Config = CFG, quiet: bool = False) -> Path:
    dest_dir = cfg.raw_dir / f"training_set{hospital}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    keys = list_keys(hospital, cfg)
    if not quiet:
        print(f"[data] hospital {hospital}: {len(keys):,} admissions listed", flush=True)

    fetched = 0
    with ThreadPoolExecutor(max_workers=cfg.download_workers) as pool:
        futures = [pool.submit(_fetch_one, k, dest_dir, cfg.s3_bucket_url) for k in keys]
        for i, fut in enumerate(as_completed(futures), 1):
            _, did = fut.result()
            fetched += did
            if not quiet and i % 2000 == 0:
                print(f"[data]   {i:,}/{len(keys):,}", flush=True)
    if not quiet:
        print(f"[data] hospital {hospital}: {fetched:,} newly downloaded", flush=True)
    return dest_dir


def _read_psv(path: Path, hospital: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", na_values=["NaN"])
    df["patient_id"] = path.stem
    df["hospital"] = hospital
    return df


def build_parquet(hospital: str, cfg: Config = CFG, quiet: bool = False) -> Path:
    """Concatenate one hospital's admissions into a single Parquet table."""
    out = cfg.interim_dir / f"set{hospital}.parquet"
    if out.exists():
        if not quiet:
            print(f"[data] {out.name} already built", flush=True)
        return out

    src_dir = cfg.raw_dir / f"training_set{hospital}"
    paths = sorted(src_dir.glob("p*.psv"))
    if not paths:
        raise FileNotFoundError(f"no .psv files in {src_dir}; run the download step first")

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, df in enumerate(pool.map(lambda p: _read_psv(p, hospital), paths), 1):
            frames.append(df)
            if not quiet and i % 5000 == 0:
                print(f"[data]   parsed {i:,}/{len(paths):,}", flush=True)

    table = pd.concat(frames, ignore_index=True)
    # ICULOS is 1-indexed in the source; keep a 0-indexed hour for window maths.
    table["hour"] = table["ICULOS"].astype("int32") - 1
    table = table.sort_values(["patient_id", "hour"], ignore_index=True)

    for col in table.columns:
        if table[col].dtype == "float64":
            table[col] = table[col].astype("float32")

    cfg.interim_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    if not quiet:
        n_pat = table["patient_id"].nunique()
        n_sep = table.groupby("patient_id")["SepsisLabel"].max().sum()
        print(
            f"[data] set{hospital}: {len(table):,} hours, {n_pat:,} admissions, "
            f"{int(n_sep):,} septic ({n_sep / n_pat:.1%})",
            flush=True,
        )
    return out


def ensure_data(hospitals=SETS, cfg: Config = CFG, quiet: bool = False) -> dict[str, Path]:
    cfg.ensure_dirs()
    paths = {}
    for h in hospitals:
        if not (cfg.interim_dir / f"set{h}.parquet").exists():
            download_hospital(h, cfg, quiet)
        paths[h] = build_parquet(h, cfg, quiet)
    return paths


if __name__ == "__main__":  # pragma: no cover
    ensure_data(sys.argv[1:] or SETS)
