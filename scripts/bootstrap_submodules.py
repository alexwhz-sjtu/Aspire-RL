#!/usr/bin/env python3
"""Initialize pinned source dependencies and restore the captured local patches."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all', action='store_true', help='Also initialize BEHAVIOR/cuRobo dependencies')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'provenance/current_snapshot.json').read_text())
    dependencies = [entry for entry in manifest['submodules'] if args.all or Path(entry['path']).name not in {'b1k', 'curobo'}]
    paths = [entry['path'] for entry in dependencies]
    subprocess.run(['git', 'submodule', 'update', '--init', '--recursive', '--', *paths], cwd=root, check=True)
    for entry in dependencies:
        target = root / entry['path']
        actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=target, text=True).strip()
        if actual != entry['commit']:
            raise RuntimeError('Dependency commit mismatch: ' + entry['path'])
        if entry.get('patch'):
            patch = root / entry['patch']
            can_apply = subprocess.run(['git', 'apply', '--check', str(patch)], cwd=target, capture_output=True).returncode == 0
            already_applied = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=target, capture_output=True).returncode == 0
            if can_apply:
                subprocess.run(['git', 'apply', str(patch)], cwd=target, check=True)
            elif not already_applied:
                raise RuntimeError('Patch conflicts with local dependency changes: ' + entry['path'])
            for relative, expected in entry['modified_file_sha256'].items():
                actual = hashlib.sha256((target / relative).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError('Patched dependency file mismatch: ' + entry['path'] + '/' + relative)
        print('Verified:', entry['path'], entry['commit'])


if __name__ == '__main__':
    main()
