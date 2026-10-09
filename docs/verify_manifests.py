#!/usr/bin/env python3
"""Verify archived file inventories, SHA-256 checksums, and file sizes."""
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Git metadata and disposable local environment/output files are not archived inputs.
EXCLUDED_DIRECTORIES = frozenset({
    ".git", ".venv", "venv", "work", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".vscode", ".idea",
})
EXCLUDED_FILENAMES = frozenset({".DS_Store"})
EXCLUDED_SUFFIXES = frozenset({".pyc", ".pyo", ".nbc", ".nbi", ".log"})


def is_archived_file(path, base):
    relative = path.relative_to(base)
    if any(part in EXCLUDED_DIRECTORIES for part in relative.parts):
        return False
    if path.name in EXCLUDED_FILENAMES or path.suffix in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(base, manifest):
    with manifest.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["path", "size_bytes", "sha256"]:
            raise ValueError(f"Unexpected manifest columns: {manifest}")
        rows = list(reader)

    listed = set()
    for row in rows:
        name = row["path"]
        relative = Path(name)
        if (not name or relative.is_absolute() or ".." in relative.parts
                or relative.as_posix() != name):
            raise ValueError(f"Invalid manifest path: {name!r}")
        if name in listed:
            raise ValueError(f"Duplicate manifest entry: {name}")
        listed.add(name)
        path = base / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest file: {path}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise ValueError(f"File size differs: {path}")
        if sha256(path) != row["sha256"]:
            raise ValueError(f"SHA-256 differs: {path}")

    actual = {
        path.relative_to(base).as_posix()
        for path in base.rglob("*")
        if path != manifest and is_archived_file(path, base)
    }
    if listed != actual:
        missing = sorted(actual - listed)
        extra = sorted(listed - actual)
        raise ValueError(
            f"File inventory differs: {manifest}; "
            f"unlisted={missing[:10]}; absent_or_excluded={extra[:10]}"
        )
    print(f"PASS {manifest.relative_to(ROOT)}: {len(rows)} files")


if __name__ == "__main__":
    verify(ROOT / "data", ROOT / "data/SHA256_DATA.csv")
    verify(ROOT / "results", ROOT / "results/SHA256_RESULTS.csv")
    verify(ROOT, ROOT / "docs/SHA256_MANIFEST.csv")
