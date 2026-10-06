"""Static campaign dataset for grouped ASPIRE rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_campaign_rows(path: str | Path) -> Any:
    from datasets import Dataset

    manifest = json.loads(Path(path).read_text())
    if manifest.get("protocol_version") != "aspire-online-rl/v1":
        raise ValueError("campaign protocol_version must be aspire-online-rl/v1")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("campaign manifest must contain tasks")
    rows = []
    for task in tasks:
        if "heldout_seeds" in task:
            raise ValueError("held-out seeds must not be present in the Actor dataset")
        task_uid = task.get("task_uid")
        if not isinstance(task_uid, str) or not task_uid:
            raise ValueError("every campaign task requires task_uid")
        rows.append({"task_uid": task_uid})
    return Dataset.from_list(rows)
