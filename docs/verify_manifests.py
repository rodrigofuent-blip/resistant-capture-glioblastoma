#!/usr/bin/env python3
"""Verify archived file inventories, SHA-256 checksums and file sizes."""
from pathlib import Path
import csv,hashlib
ROOT=Path(__file__).resolve().parents[1]
def verify(base,manifest):
    rows=list(csv.DictReader(manifest.open(newline='',encoding='utf-8')))
    paths=set()
    for row in rows:
        name=row['path']
        assert name not in paths, f'Duplicate entry: {name}'
        paths.add(name)
        p=base/name
        assert p.is_file(), f'Missing file {p}'
        assert int(row['size_bytes'])==p.stat().st_size, f'Size differs {p}'
        assert row['sha256']==hashlib.sha256(p.read_bytes()).hexdigest(),f'Checksum differs {p}'
    actual={p.relative_to(base).as_posix() for p in base.rglob('*')
            if p.is_file() and p!=manifest and 'work' not in p.parts and '.venv' not in p.parts
            and '__pycache__' not in p.parts and p.suffix not in ('.nbc','.nbi','.pyc')}
    assert paths==actual, f'Inventory differs {manifest}: missing={sorted(actual-paths)[:5]}, extra={sorted(paths-actual)[:5]}'
    print(f'PASS {manifest.relative_to(ROOT)}: {len(rows)} files')
if __name__=='__main__':
    verify(ROOT/'data',ROOT/'data/SHA256_DATA.csv')
    verify(ROOT/'results',ROOT/'results/SHA256_RESULTS.csv')
    verify(ROOT,ROOT/'docs/SHA256_MANIFEST.csv')
