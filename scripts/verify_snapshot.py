#!/usr/bin/env python3
"""Verify captured source bytes without importing GPU or simulator packages."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/current_snapshot.json').read_text())
    errors = []
    for relative, expected in manifest['files'].items():
        path = root / relative
        if expected.get('symlink') is not None:
            if not path.is_symlink() or str(path.readlink()) != expected['symlink']:
                errors.append(relative)
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
            errors.append(relative)
    if errors:
        raise SystemExit('Snapshot differs from captured baseline:\n' + '\n'.join(errors))
    print('Verified', len(manifest['files']), 'captured files against SHA256; dependency pins are recorded in provenance.')


if __name__ == '__main__':
    main()
