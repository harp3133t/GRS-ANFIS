#!/usr/bin/env python3
"""Download external dataset snapshots for the GRS-ANFIS notebooks."""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from ucimlrepo import fetch_ucirepo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, source: str, source_url: str, status: str) -> dict:
    return {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "source": source,
        "source_url": source_url,
        "status": status,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_columns_file(csv_path: Path) -> Path:
    columns_path = csv_path.with_name(csv_path.stem + "_columns.txt")
    columns = pd.read_csv(csv_path, nrows=0).columns
    columns_path.write_text("\n".join(columns), encoding="utf-8")
    return columns_path


def save_openml_snapshot(data_id: int, out_name: str, overwrite: bool) -> list[dict]:
    out_path = DATA_DIR / out_name
    source = f"OpenML data_id={data_id}"
    fallback_url = f"https://www.openml.org/d/{data_id}"
    if out_path.exists() and not overwrite:
        print(f"skip existing {out_path.relative_to(PROJECT_ROOT)}")
        return [file_record(out_path, source=source, source_url=fallback_url, status="existing")]

    dataset = fetch_openml(data_id=data_id, as_frame=True)
    df = dataset.data.copy()
    df["target"] = dataset.target.reset_index(drop=True)
    df.to_csv(out_path, index=False)
    source_url = dataset.details.get("url") or fallback_url
    return [file_record(out_path, source=source, source_url=source_url, status="downloaded")]


def save_ucirepo_snapshot(dataset_id: int, out_name: str, overwrite: bool) -> list[dict]:
    out_path = DATA_DIR / out_name
    columns_path = out_path.with_name(out_path.stem + "_columns.txt")
    source = f"UCI id={dataset_id}"
    fallback_url = f"https://archive.ics.uci.edu/dataset/{dataset_id}"
    if out_path.exists() and columns_path.exists() and not overwrite:
        print(f"skip existing {out_path.relative_to(PROJECT_ROOT)}")
        return [
            file_record(out_path, source=source, source_url=fallback_url, status="existing"),
            file_record(columns_path, source=source, source_url=fallback_url, status="existing"),
        ]

    dataset = fetch_ucirepo(id=dataset_id)
    X = dataset.data.features.copy().reset_index(drop=True)
    y = dataset.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError(f"Expected one target column for UCI id={dataset_id}, got {list(y.columns)}")
        y = y.iloc[:, 0]
    y = pd.Series(y, name="target").reset_index(drop=True)
    df = pd.concat([X, y], axis=1)
    df.to_csv(out_path, index=False)
    columns_path = write_columns_file(out_path)
    source_url = dataset.metadata.get("repository_url") or fallback_url
    return [
        file_record(out_path, source=source, source_url=source_url, status="downloaded"),
        file_record(columns_path, source=source, source_url=source_url, status="generated"),
    ]


def download_url(urls: str | list[str], out_name: str, source: str, overwrite: bool) -> dict:
    if isinstance(urls, str):
        urls = [urls]
    out_path = DATA_DIR / out_name
    if out_path.exists() and not overwrite:
        print(f"skip existing {out_path.relative_to(PROJECT_ROOT)}")
        return file_record(out_path, source=source, source_url=urls[0], status="existing")

    last_error = None
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    for url in urls:
        try:
            print(f"download {url} -> {out_path.relative_to(PROJECT_ROOT)}")
            with urllib.request.urlopen(url, timeout=60) as response, tmp_path.open("wb") as out_fh:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    out_fh.write(chunk)
            tmp_path.replace(out_path)
            return file_record(out_path, source=source, source_url=url, status="downloaded")
        except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            print(f"failed {url}: {exc}", file=sys.stderr)
    raise RuntimeError(f"Could not download {out_name}") from last_error


def text_matrix_shape(path: Path) -> tuple[int, int]:
    rows = 0
    cols = None
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            rows += 1
            if cols is None:
                cols = len(stripped.split())
    return rows, int(cols or 0)


def validate_snapshot_shapes() -> pd.DataFrame:
    checks = []
    for name, expected in {
        "vowel_openml_307.csv": (990, 13),
        "spambase_uci_94.csv": (4601, 58),
        "bcwd_uci_15.csv": (699, 10),
    }.items():
        df = pd.read_csv(DATA_DIR / name)
        checks.append({"dataset_file": name, "shape": tuple(df.shape), "expected": expected, "ok": tuple(df.shape) == expected})

    for name, expected in {
        "gisette_train.data": (6000, 5000),
        "gisette_valid.data": (1000, 5000),
        "gisette_test.data": (6500, 5000),
        "gisette_train.labels": (6000, 1),
    }.items():
        shape = text_matrix_shape(DATA_DIR / name)
        checks.append({"dataset_file": name, "shape": shape, "expected": expected, "ok": shape == expected})
    return pd.DataFrame(checks)


def validate_project_loaders() -> pd.DataFrame:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from data import load_bcwd_data, load_gisette_data, load_spambase_data, load_vowel_data

    checks = []
    for name, loader, expected in [
        ("Vowel", load_vowel_data, (990, 29)),
        ("Spambase", load_spambase_data, (4601, 57)),
        ("Breast Cancer Wisconsin (Original)", load_bcwd_data, (699, 80)),
        ("Gisette", load_gisette_data, (6000, 5000)),
    ]:
        X, y, feature_names = loader()
        shape = tuple(X.shape)
        checks.append(
            {
                "dataset": name,
                "X_shape": shape,
                "y_len": len(y),
                "n_features": len(feature_names),
                "expected_X_shape": expected,
                "ok": shape == expected and len(y) == expected[0],
            }
        )
    return pd.DataFrame(checks)


def download_all(overwrite: bool) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    # This local snapshot matches OpenML data id 307. The OpenML record cites
    # the original UCI Connectionist Bench Vowel Recognition source.
    records.extend(save_openml_snapshot(307, "vowel_openml_307.csv", overwrite))

    # UCI snapshots used directly by data.py.
    records.extend(save_ucirepo_snapshot(94, "spambase_uci_94.csv", overwrite))
    records.extend(save_ucirepo_snapshot(15, "bcwd_uci_15.csv", overwrite))

    # Gisette is stored as raw UCI archive files because data.py reads the
    # original whitespace-delimited matrices directly.
    gisette_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/gisette"
    records.append(download_url(f"{gisette_base}/GISETTE/gisette_train.data", "gisette_train.data", "UCI id=170 raw archive", overwrite))
    records.append(download_url(f"{gisette_base}/GISETTE/gisette_train.labels", "gisette_train.labels", "UCI id=170 raw archive", overwrite))
    records.append(download_url(f"{gisette_base}/GISETTE/gisette_valid.data", "gisette_valid.data", "UCI id=170 raw archive", overwrite))
    records.append(download_url(f"{gisette_base}/GISETTE/gisette_test.data", "gisette_test.data", "UCI id=170 raw archive", overwrite))
    records.append(download_url([f"{gisette_base}/gisette.param", f"{gisette_base}/GISETTE/gisette.param"], "gisette.param", "UCI id=170 raw archive", overwrite))

    manifest = pd.DataFrame(records)
    manifest_path = DATA_DIR / "data_source_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"wrote {manifest_path.relative_to(PROJECT_ROOT)}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Refresh files even when local snapshots already exist.")
    parser.add_argument("--skip-loader-checks", action="store_true", help="Only validate raw file shapes, not data.py loader output.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"DATA_DIR={DATA_DIR}")

    manifest = download_all(overwrite=args.overwrite)
    print(manifest[["file", "bytes", "status", "source"]].to_string(index=False))

    raw_checks = validate_snapshot_shapes()
    print("\nRaw snapshot checks")
    print(raw_checks.to_string(index=False))
    if not raw_checks["ok"].all():
        raise SystemExit("Raw snapshot validation failed")

    if not args.skip_loader_checks:
        loader_checks = validate_project_loaders()
        print("\nProject loader checks")
        print(loader_checks.to_string(index=False))
        if not loader_checks["ok"].all():
            raise SystemExit("Project loader validation failed")


if __name__ == "__main__":
    main()
