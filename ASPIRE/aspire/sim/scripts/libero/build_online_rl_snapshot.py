#!/usr/bin/env python3
"""Build one immutable L0 snapshot and Goal-Swap campaign manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from aspire.sim.online_rl.contracts import (
    PROTOCOL_VERSION,
    CampaignManifest,
    CampaignTask,
    EpisodeBudgets,
)
from aspire.sim.online_rl.snapshots import (
    actor_prompt_digest,
    build_snapshot,
    canonical_json_digest,
    online_rl_protocol_digest,
    processor_digest,
    sha256_file,
    tool_schema_digest,
)


def git_commit(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def libero_tasks(suite_name: str) -> list[tuple[int, str, str]]:
    from libero import benchmark as libero_benchmark

    benchmarks = libero_benchmark.get_benchmark_dict()
    if suite_name not in benchmarks:
        raise ValueError(f"unknown LIBERO suite: {suite_name}")
    suite = benchmarks[suite_name]()
    tasks = []
    for task_id in range(suite.n_tasks):
        task = suite.get_task(task_id)
        language = getattr(task, "language", None) or task.name.replace("_", " ")
        tasks.append((task_id, task.name, str(language)))
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-source", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--api-source", type=Path, required=True)
    parser.add_argument("--api-snapshot-dir", type=Path, required=True)
    parser.add_argument("--system-prompt", type=Path, required=True)
    parser.add_argument("--findings-prompt", type=Path, required=True)
    parser.add_argument("--tool-schema", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--areal-commit", required=True)
    parser.add_argument("--suite", default="libero_goal_swap")
    args = parser.parse_args()

    snapshot = build_snapshot(args.skill_source, args.snapshot_dir)
    api_snapshot = build_snapshot(args.api_source, args.api_snapshot_dir)
    env_config = args.env_config.resolve()
    prompt_digest = actor_prompt_digest(args.system_prompt, args.findings_prompt)
    schema_digest = tool_schema_digest(args.tool_schema)
    model_processor_digest = processor_digest(args.model_path)
    protocol_digest = online_rl_protocol_digest(
        protocol_version=PROTOCOL_VERSION,
        budgets=EpisodeBudgets().model_dump(mode="json"),
        skill_snapshot_digest=snapshot["aggregate_sha256"],
        api_snapshot_digest=api_snapshot["aggregate_sha256"],
        prompt_digest=prompt_digest,
        schema_digest=schema_digest,
        model_path=str(args.model_path.resolve()),
        model_processor_digest=model_processor_digest,
    )
    repository_root = Path(__file__).resolve().parents[4]
    tasks = [
        CampaignTask(
            task_uid=f"{args.suite}/{task_name}",
            suite=args.suite,
            task_name=task_name,
            task_id=task_id,
            task_language=language,
            env_config=str(env_config),
            env_config_digest=sha256_file(env_config),
            skill_snapshot_id=snapshot["snapshot_id"],
            skill_snapshot=str(args.snapshot_dir.resolve()),
            skill_snapshot_digest=snapshot["aggregate_sha256"],
            protocol_digest=protocol_digest,
        )
        for task_id, task_name, language in libero_tasks(args.suite)
    ]
    aspire_commit = git_commit(repository_root)
    if aspire_commit is None:
        raise RuntimeError("could not resolve the ASPIRE git commit")
    manifest = CampaignManifest(
        campaign_id=canonical_json_digest(
            {"protocol_digest": protocol_digest, "tasks": [task.task_uid for task in tasks]}
        )[:16],
        protocol_digest=protocol_digest,
        api_snapshot=str(args.api_snapshot_dir.resolve()),
        api_snapshot_digest=api_snapshot["aggregate_sha256"],
        actor_prompt_digest=prompt_digest,
        tool_schema_digest=schema_digest,
        model_path=str(args.model_path.resolve()),
        processor_digest=model_processor_digest,
        aspire_commit=aspire_commit,
        areal_commit=args.areal_commit,
        tasks=tasks,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n")
    temporary.replace(args.manifest)
    print(json.dumps({"manifest": str(args.manifest), "tasks": len(tasks)}, indent=2))


if __name__ == "__main__":
    main()
