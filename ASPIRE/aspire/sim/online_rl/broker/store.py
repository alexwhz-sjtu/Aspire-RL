"""Durable SQLite state for the ASPIRE online-RL broker."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aspire.sim.online_rl.contracts import (
    ActionResponse,
    ArtifactRef,
    EpisodeCounters,
    EpisodeState,
    Observation,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()  # noqa: UP017 - Python 3.10


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: str
    client_episode_id: str
    task_uid: str
    state: EpisodeState
    state_version: int
    observation: Observation
    counters: EpisodeCounters
    current_code_sha256: str | None
    frozen_code_sha256: str | None
    findings: str | None
    created_at: str
    updated_at: str


class EpisodeStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    episode_id TEXT PRIMARY KEY,
                    client_episode_id TEXT UNIQUE NOT NULL,
                    task_uid TEXT NOT NULL,
                    state TEXT NOT NULL,
                    state_version INTEGER NOT NULL,
                    observation_json TEXT NOT NULL,
                    counters_json TEXT NOT NULL,
                    current_code_sha256 TEXT,
                    frozen_code_sha256 TEXT,
                    findings TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS code_revisions (
                    episode_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    code TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (episode_id, revision),
                    UNIQUE (episode_id, sha256),
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS actions (
                    episode_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (episode_id, request_id),
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS replays (
                    replay_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    code_sha256 TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    terminal INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    episode_id TEXT NOT NULL,
                    replay_id TEXT,
                    sha256 TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS terminal_scores (
                    episode_id TEXT PRIMARY KEY,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    event_class TEXT NOT NULL,
                    code TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
                );
                """
            )
            action_columns = {
                row["name"]
                for row in self._connection.execute("PRAGMA table_info(actions)")
            }
            if "request_json" not in action_columns:
                self._connection.execute(
                    "ALTER TABLE actions ADD COLUMN request_json TEXT NOT NULL DEFAULT '{}'"
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def fail_orphaned_episodes(self, *, stale_after_seconds: int) -> int:
        """Mark stale, non-terminal episodes as infrastructure failures.

        Recovery remains the default: callers must explicitly enable this startup
        cleanup after choosing a threshold longer than the maximum reconnect
        window.  Completed, policy-failed, infrastructure-failed, and explicitly
        closed episodes are immutable and never selected.
        """
        if stale_after_seconds < 1:
            raise ValueError("stale_after_seconds must be positive")
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=stale_after_seconds)  # noqa: UP017
        terminal_states = {
            EpisodeState.COMPLETED,
            EpisodeState.POLICY_FAILED,
            EpisodeState.INFRA_FAILED,
            EpisodeState.CLOSED,
        }
        with self._lock, self._connection:
            rows = self._connection.execute(
                "SELECT episode_id, state, state_version, updated_at FROM episodes"
            ).fetchall()
            orphaned = []
            for row in rows:
                state = EpisodeState(row["state"])
                if state in terminal_states:
                    continue
                updated_at = datetime.fromisoformat(row["updated_at"])
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)  # noqa: UP017
                if updated_at < cutoff:
                    orphaned.append(row)
            now = utc_now()
            for row in orphaned:
                cursor = self._connection.execute(
                    """
                    UPDATE episodes
                    SET state=?, state_version=state_version+1, updated_at=?
                    WHERE episode_id=? AND state_version=?
                    """,
                    (
                        EpisodeState.INFRA_FAILED.value,
                        now,
                        row["episode_id"],
                        row["state_version"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("concurrent orphan episode cleanup")
                self._connection.execute(
                    """
                    INSERT INTO events (
                        episode_id, event_class, code, message, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        row["episode_id"],
                        "infrastructure",
                        "orphan_cleanup",
                        (
                            "Broker startup marked a stale episode as orphaned; "
                            f"previous_state={row['state']}"
                        ),
                        now,
                    ),
                )
        return len(orphaned)

    def create_episode(
        self,
        *,
        episode_id: str,
        client_episode_id: str,
        task_uid: str,
        observation: Observation,
    ) -> EpisodeRecord:
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO episodes (
                    episode_id, client_episode_id, task_uid, state, state_version,
                    observation_json, counters_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    client_episode_id,
                    task_uid,
                    EpisodeState.CONTEXT_READY.value,
                    0,
                    observation.model_dump_json(),
                    EpisodeCounters().model_dump_json(),
                    now,
                    now,
                ),
            )
        return self.get_episode(episode_id)

    def get_episode(self, episode_id: str) -> EpisodeRecord:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"episode not found: {episode_id}")
        return EpisodeRecord(
            episode_id=row["episode_id"],
            client_episode_id=row["client_episode_id"],
            task_uid=row["task_uid"],
            state=EpisodeState(row["state"]),
            state_version=int(row["state_version"]),
            observation=Observation.model_validate_json(row["observation_json"]),
            counters=EpisodeCounters.model_validate_json(row["counters_json"]),
            current_code_sha256=row["current_code_sha256"],
            frozen_code_sha256=row["frozen_code_sha256"],
            findings=row["findings"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def find_by_client_episode_id(self, client_episode_id: str) -> EpisodeRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT episode_id FROM episodes WHERE client_episode_id = ?",
                (client_episode_id,),
            ).fetchone()
        return self.get_episode(row["episode_id"]) if row is not None else None

    def update_episode(
        self,
        episode_id: str,
        *,
        expected_version: int,
        state: EpisodeState | None = None,
        counters: EpisodeCounters | None = None,
        current_code_sha256: str | None = None,
        frozen_code_sha256: str | None = None,
        findings: str | None = None,
        set_findings: bool = False,
    ) -> EpisodeRecord:
        current = self.get_episode(episode_id)
        if current.state_version != expected_version:
            raise RuntimeError(
                f"state version conflict: expected {expected_version}, "
                f"current {current.state_version}"
            )
        values = {
            "state": (state or current.state).value,
            "state_version": current.state_version + 1,
            "counters_json": (counters or current.counters).model_dump_json(),
            "current_code_sha256": (
                current.current_code_sha256
                if current_code_sha256 is None
                else current_code_sha256
            ),
            "frozen_code_sha256": (
                current.frozen_code_sha256
                if frozen_code_sha256 is None
                else frozen_code_sha256
            ),
            "findings": findings if set_findings else current.findings,
            "updated_at": utc_now(),
        }
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE episodes SET state=:state, state_version=:state_version,
                    counters_json=:counters_json,
                    current_code_sha256=:current_code_sha256,
                    frozen_code_sha256=:frozen_code_sha256,
                    findings=:findings, updated_at=:updated_at
                WHERE episode_id=:episode_id AND state_version=:expected_version
                """,
                {**values, "episode_id": episode_id, "expected_version": expected_version},
            )
            if cursor.rowcount != 1:
                raise RuntimeError("concurrent episode state update")
        return self.get_episode(episode_id)

    def add_code_revision(self, episode_id: str, sha256: str, code: str) -> int:
        with self._lock, self._connection:
            existing = self._connection.execute(
                "SELECT revision FROM code_revisions WHERE episode_id=? AND sha256=?",
                (episode_id, sha256),
            ).fetchone()
            if existing is not None:
                return int(existing["revision"])
            row = self._connection.execute(
                "SELECT COALESCE(MAX(revision), 0) + 1 AS revision "
                "FROM code_revisions WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
            revision = int(row["revision"])
            self._connection.execute(
                "INSERT INTO code_revisions VALUES (?, ?, ?, ?, ?)",
                (episode_id, revision, sha256, code, utc_now()),
            )
            return revision

    def get_code(self, episode_id: str, sha256: str) -> str:
        with self._lock:
            row = self._connection.execute(
                "SELECT code FROM code_revisions WHERE episode_id=? AND sha256=?",
                (episode_id, sha256),
            ).fetchone()
        if row is None:
            raise KeyError("code hash does not belong to this episode")
        return str(row["code"])

    def get_action(self, episode_id: str, request_id: str) -> ActionResponse | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM actions WHERE episode_id=? AND request_id=?",
                (episode_id, request_id),
            ).fetchone()
        return ActionResponse.model_validate_json(row["response_json"]) if row else None

    def record_action(
        self,
        episode_id: str,
        request_id: str,
        request: dict[str, Any],
        response: ActionResponse,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO actions (
                    episode_id, request_id, request_json, response_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    episode_id,
                    request_id,
                    json.dumps(request, sort_keys=True),
                    response.model_dump_json(),
                    utc_now(),
                ),
            )

    def get_action_request(self, episode_id: str, request_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT request_json FROM actions WHERE episode_id=? AND request_id=?",
                (episode_id, request_id),
            ).fetchone()
        if row is None:
            raise KeyError("action request not found")
        return json.loads(row["request_json"])

    def metrics_snapshot(self) -> dict[str, Any]:
        with self._lock:
            episode_rows = self._connection.execute(
                "SELECT state, COUNT(*) AS count FROM episodes GROUP BY state"
            ).fetchall()
            action_rows = self._connection.execute(
                "SELECT request_json, response_json FROM actions"
            ).fetchall()
            replay_rows = self._connection.execute(
                "SELECT terminal, COUNT(*) AS count FROM replays GROUP BY terminal"
            ).fetchall()
            score_rows = self._connection.execute(
                "SELECT response_json FROM terminal_scores"
            ).fetchall()
            event_rows = self._connection.execute(
                "SELECT event_class, code, COUNT(*) AS count FROM events "
                "GROUP BY event_class, code"
            ).fetchall()
        invalid_actions = 0
        max_turns: list[int] = []
        output_tokens: list[int] = []
        for row in action_rows:
            request = json.loads(row["request_json"])
            response = json.loads(row["response_json"])
            invalid_actions += int(not response.get("ok", False))
            if request.get("model_turn") is not None:
                max_turns.append(int(request["model_turn"]))
            if request.get("cumulative_policy_output_tokens") is not None:
                output_tokens.append(int(request["cumulative_policy_output_tokens"]))
        rewards = [float(json.loads(row["response_json"])["reward"]) for row in score_rows]
        return {
            "episodes_by_state": {
                str(row["state"]): int(row["count"]) for row in episode_rows
            },
            "actions_total": len(action_rows),
            "invalid_actions_total": invalid_actions,
            "invalid_tool_rate": invalid_actions / len(action_rows) if action_rows else 0.0,
            "max_observed_model_turn": max(max_turns, default=0),
            "max_observed_policy_output_tokens": max(output_tokens, default=0),
            "debug_replays_total": sum(
                int(row["count"]) for row in replay_rows if not bool(row["terminal"])
            ),
            "terminal_replays_total": sum(
                int(row["count"]) for row in replay_rows if bool(row["terminal"])
            ),
            "terminal_episodes_total": len(rewards),
            "terminal_reward_mean": sum(rewards) / len(rewards) if rewards else 0.0,
            "events": {
                f"{row['event_class']}:{row['code']}": int(row["count"])
                for row in event_rows
            },
        }

    def record_event(
        self,
        episode_id: str,
        *,
        event_class: str,
        code: str,
        message: str,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO events (
                    episode_id, event_class, code, message, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (episode_id, event_class, code, message[:4000], utc_now()),
            )

    def add_replay(
        self,
        *,
        replay_id: str,
        episode_id: str,
        seed: int,
        code_sha256: str,
        result: dict[str, Any],
        terminal: bool,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO replays VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    replay_id,
                    episode_id,
                    seed,
                    code_sha256,
                    json.dumps(result, sort_keys=True),
                    int(terminal),
                    utc_now(),
                ),
            )

    def get_replay(self, episode_id: str, replay_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._connection.execute(
                "SELECT result_json FROM replays WHERE episode_id=? AND replay_id=?",
                (episode_id, replay_id),
            ).fetchone()
        if row is None:
            raise KeyError("replay does not belong to this episode")
        return json.loads(row["result_json"])

    def add_artifact(
        self,
        episode_id: str,
        artifact: ArtifactRef,
        *,
        replay_id: str | None = None,
    ) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    artifact.artifact_id,
                    episode_id,
                    replay_id,
                    artifact.sha256,
                    artifact.media_type,
                    artifact.size_bytes,
                    artifact.kind,
                    utc_now(),
                ),
            )

    def get_artifact(self, episode_id: str, artifact_id: str) -> ArtifactRef:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE episode_id=? AND artifact_id=?",
                (episode_id, artifact_id),
            ).fetchone()
        if row is None:
            raise KeyError("artifact does not belong to this episode")
        return ArtifactRef(
            artifact_id=row["artifact_id"],
            sha256=row["sha256"],
            media_type=row["media_type"],
            size_bytes=int(row["size_bytes"]),
            kind=row["kind"],
        )

    def list_replay_artifacts(self, episode_id: str, replay_id: str) -> list[ArtifactRef]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM artifacts WHERE episode_id=? AND replay_id=? "
                "ORDER BY created_at, artifact_id",
                (episode_id, replay_id),
            ).fetchall()
        return [
            ArtifactRef(
                artifact_id=row["artifact_id"],
                sha256=row["sha256"],
                media_type=row["media_type"],
                size_bytes=int(row["size_bytes"]),
                kind=row["kind"],
            )
            for row in rows
        ]

    def get_terminal_score(self, episode_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT response_json FROM terminal_scores WHERE episode_id=?",
                (episode_id,),
            ).fetchone()
        return json.loads(row["response_json"]) if row else None

    def complete_terminal_score(
        self,
        episode_id: str,
        *,
        expected_version: int,
        state: EpisodeState,
        response: dict[str, Any],
    ) -> EpisodeRecord:
        """Atomically persist the terminal result and final episode state."""
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO terminal_scores VALUES (?, ?, ?)",
                (episode_id, json.dumps(response, sort_keys=True), now),
            )
            cursor = self._connection.execute(
                """
                UPDATE episodes SET state=?, state_version=state_version+1,
                    updated_at=? WHERE episode_id=? AND state_version=?
                """,
                (state.value, now, episode_id, expected_version),
            )
            if cursor.rowcount != 1:
                raise RuntimeError("concurrent terminal score update")
        return self.get_episode(episode_id)
