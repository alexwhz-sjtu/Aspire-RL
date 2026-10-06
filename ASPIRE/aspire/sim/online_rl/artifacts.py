"""Content-addressed artifact storage owned by the ASPIRE broker."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from aspire.sim.online_rl.contracts import ArtifactRef


class ArtifactStore:
    """Persist immutable bytes by digest; authorization is handled by EpisodeStore."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, *, artifact_id: str, kind: str, media_type: str, data: bytes) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        target = self.root / digest[:2] / digest
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
        return ArtifactRef(
            artifact_id=artifact_id,
            sha256=digest,
            media_type=media_type,
            size_bytes=len(data),
            kind=kind,
        )

    def read(self, sha256: str) -> bytes:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("invalid artifact sha256")
        return (self.root / sha256[:2] / sha256).read_bytes()
