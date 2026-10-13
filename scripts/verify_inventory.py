"""Verify the published project snapshot using only the Python standard library."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = root / 'docs/inventory/files.csv'
    errors, verified, excluded = [], 0, 0
    with manifest.open(newline='') as source:
        for row in csv.DictReader(source):
            relative = Path(row['path'])
            if relative.is_absolute() or '..' in relative.parts:
                errors.append(f"Unsafe manifest path: {row['path']}")
                continue
            path = root / relative
            if row['status'] == 'excluded':
                excluded += 1
                if path.exists() or path.is_symlink():
                    errors.append(f"Excluded file present: {row['path']}")
                continue
            if row['kind'] == 'symlink':
                if not path.is_symlink() or path.readlink().as_posix() != row['link_target']:
                    errors.append(f"Symlink mismatch: {row['path']}")
                else:
                    verified += 1
                continue
            if not path.is_file():
                errors.append(f"Missing file: {row['path']}")
                continue
            if path.stat().st_size != int(row['bytes']):
                errors.append(f"Size mismatch: {row['path']} (run git lfs pull for LFS resources)")
                continue
            if file_hash(path) != row['sha256']:
                errors.append(f"SHA-256 mismatch: {row['path']}")
                continue
            verified += 1
    if errors:
        for error in errors[:30]:
            print(error, file=sys.stderr)
        print(f'FAILED: {len(errors)} mismatches', file=sys.stderr)
        return 1
    print(f'OK: {verified} uploaded project files verified; {excluded} excluded files absent.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
