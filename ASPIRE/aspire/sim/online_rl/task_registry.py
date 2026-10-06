"""Campaign-backed LIBERO task registry."""

from __future__ import annotations

import json
from pathlib import Path

from aspire.sim.online_rl.contracts import CampaignManifest, CampaignTask
from aspire.sim.online_rl.snapshots import (
    online_rl_protocol_digest,
    sha256_file,
    verify_snapshot,
)


class TaskRegistry:
    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = CampaignManifest.model_validate_json(self.manifest_path.read_text())
        self._tasks = {task.task_uid: task for task in self.manifest.tasks}
        if len(self._tasks) != len(self.manifest.tasks):
            raise ValueError("campaign manifest contains duplicate task_uid values")
        skill_digests = {task.skill_snapshot_digest for task in self.manifest.tasks}
        budget_payloads = {
            task.budgets.model_dump_json() for task in self.manifest.tasks
        }
        if len(skill_digests) != 1 or len(budget_payloads) != 1:
            raise ValueError("V1 campaign tasks must share one skill snapshot and budget")
        expected_protocol_digest = online_rl_protocol_digest(
            protocol_version=self.manifest.protocol_version,
            budgets=self.manifest.tasks[0].budgets.model_dump(mode="json"),
            skill_snapshot_digest=next(iter(skill_digests)),
            api_snapshot_digest=self.manifest.api_snapshot_digest,
            prompt_digest=self.manifest.actor_prompt_digest,
            schema_digest=self.manifest.tool_schema_digest,
            model_path=self.manifest.model_path,
            model_processor_digest=self.manifest.processor_digest,
        )
        if self.manifest.protocol_digest != expected_protocol_digest:
            raise ValueError(
                "campaign protocol digest does not match its component snapshots"
            )
        verify_snapshot(
            self.resolve_api_snapshot(), self.manifest.api_snapshot_digest
        )
        for task in self.manifest.tasks:
            self.verify_task_sources(task, verify_api=False)

    def verify_task_sources(
        self, task: CampaignTask, *, verify_api: bool = True
    ) -> None:
        if task.protocol_digest != self.manifest.protocol_digest:
            raise ValueError(
                f"protocol digest mismatch for {task.task_uid}: "
                f"{task.protocol_digest} != {self.manifest.protocol_digest}"
            )
        if verify_api:
            verify_snapshot(
                self.resolve_api_snapshot(), self.manifest.api_snapshot_digest
            )
        config_path = self.resolve_env_config(task)
        actual = sha256_file(config_path)
        if actual != task.env_config_digest:
            raise ValueError(
                f"environment config digest mismatch for {task.task_uid}: "
                f"expected {task.env_config_digest}, got {actual}"
            )
        verify_snapshot(
            self.resolve_skill_snapshot(task), task.skill_snapshot_digest
        )

    def get(self, task_uid: str) -> CampaignTask:
        try:
            return self._tasks[task_uid]
        except KeyError as exc:
            raise KeyError(f"task_uid is not present in campaign: {task_uid}") from exc

    def resolve_env_config(self, task: CampaignTask) -> Path:
        path = Path(task.env_config)
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise ValueError(f"environment config does not exist: {path}")
        return path

    def resolve_skill_snapshot(self, task: CampaignTask) -> Path:
        path = Path(task.skill_snapshot)
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"skill snapshot does not exist: {path}")
        return path

    def resolve_api_snapshot(self) -> Path:
        path = Path(self.manifest.api_snapshot)
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"API snapshot does not exist: {path}")
        return path

    def verify_api_snapshot_path(self, path: str | Path) -> Path:
        candidate = Path(path).resolve()
        verify_snapshot(candidate, self.manifest.api_snapshot_digest)
        return candidate

    def to_json(self) -> str:
        return json.dumps(self.manifest.model_dump(mode="json"), sort_keys=True)
