"""Build and verify immutable ASPIRE online-RL source snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from typing import Any

PROCESSOR_METADATA_FILES = (
    "chat_template.jinja",
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def processor_digest(model_path: str | Path) -> str:
    root = Path(model_path).resolve()
    entries = []
    for name in PROCESSOR_METADATA_FILES:
        path = root / name
        if path.is_file():
            entries.append(
                {
                    "path": name,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    if not entries:
        raise ValueError(f"no processor/tokenizer metadata found under model path: {root}")
    return canonical_json_digest(entries)


def actor_prompt_digest(system_prompt: str | Path, findings_prompt: str | Path) -> str:
    return canonical_json_digest(
        {
            "findings.md": Path(findings_prompt).read_text(),
            "system.md": Path(system_prompt).read_text(),
        }
    )


def tool_schema_digest(schema_path: str | Path) -> str:
    return canonical_json_digest(json.loads(Path(schema_path).read_text()))


def online_rl_protocol_digest(
    *,
    protocol_version: str,
    budgets: dict[str, Any],
    skill_snapshot_digest: str,
    api_snapshot_digest: str,
    prompt_digest: str,
    schema_digest: str,
    model_path: str,
    model_processor_digest: str,
) -> str:
    return canonical_json_digest(
        {
            "protocol_version": protocol_version,
            "development_seeds": list(range(51, 66)),
            "reward": "max(env_reward, 0.1 * (sandbox_rc == 0))",
            "budgets": budgets,
            "skill_snapshot_digest": skill_snapshot_digest,
            "api_snapshot_digest": api_snapshot_digest,
            "actor_prompt_digest": prompt_digest,
            "tool_schema_digest": schema_digest,
            "model_path": model_path,
            "processor_digest": model_processor_digest,
        }
    )


def _snapshot_files(source: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"snapshot source must not contain symlinks: {path}")
        if path.is_file():
            files.append(path)
    if not files:
        raise ValueError(f"snapshot source contains no files: {source}")
    return files


def build_snapshot(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Copy approved content once and write a digest manifest.

    ``destination`` must not already exist so a campaign cannot silently replace
    the source used by running episodes.
    """
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if not source_path.is_dir():
        raise ValueError(f"snapshot source is not a directory: {source_path}")
    if destination_path.exists():
        raise FileExistsError(f"snapshot destination already exists: {destination_path}")

    files = _snapshot_files(source_path)
    entries = [
        {
            "path": PurePosixPath(path.relative_to(source_path)).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in files
    ]
    aggregate = canonical_json_digest(entries)
    destination_path.mkdir(parents=True)
    for entry, source_file in zip(entries, files, strict=True):
        target = destination_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_file, target)
        target.chmod(0o444)
    manifest = {
        "schema_version": 1,
        "snapshot_id": aggregate[:16],
        "aggregate_sha256": aggregate,
        "files": entries,
    }
    manifest_path = destination_path / "snapshot_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_path.chmod(0o444)
    destination_path.chmod(0o555)
    return manifest


def verify_snapshot(path: str | Path, expected_digest: str | None = None) -> dict[str, Any]:
    snapshot_path = Path(path).resolve()
    manifest = json.loads((snapshot_path / "snapshot_manifest.json").read_text())
    entries = manifest.get("files", [])
    expected_paths = {PurePosixPath(entry["path"]).as_posix() for entry in entries}
    actual_paths = set()
    for file_path in snapshot_path.rglob("*"):
        if file_path.is_symlink():
            raise ValueError(f"snapshot must not contain symlinks: {file_path}")
        if file_path.is_file() and file_path.name != "snapshot_manifest.json":
            actual_paths.add(PurePosixPath(file_path.relative_to(snapshot_path)).as_posix())
    if actual_paths != expected_paths:
        raise ValueError(
            "snapshot file set differs from manifest: "
            f"missing={sorted(expected_paths - actual_paths)}, "
            f"extra={sorted(actual_paths - expected_paths)}"
        )
    for entry in entries:
        relative = PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid path in snapshot manifest: {relative}")
        file_path = snapshot_path / relative
        if file_path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"snapshot file size mismatch: {relative}")
        if sha256_file(file_path) != entry["sha256"]:
            raise ValueError(f"snapshot file digest mismatch: {relative}")
    aggregate = canonical_json_digest(entries)
    if aggregate != manifest.get("aggregate_sha256"):
        raise ValueError("snapshot aggregate digest mismatch")
    if expected_digest is not None and aggregate != expected_digest:
        raise ValueError(
            f"snapshot digest mismatch: expected {expected_digest}, got {aggregate}"
        )
    return manifest
