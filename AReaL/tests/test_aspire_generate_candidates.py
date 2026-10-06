# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import pytest

from examples.aspire.generate_candidates import (
    _load_candidates,
    _validate_existing_candidate,
    _write_candidates,
)


def test_candidate_checkpoint_is_resumable_and_sha_verified(tmp_path: Path) -> None:
    code = "print('robot')\n"
    code_path = tmp_path / "fix_code.py"
    code_path.write_text(code)
    task_uid = "libero_goal_swap/test_task"
    candidates = {
        task_uid: {
            "episode_id": "episode",
            "reward": 0.1,
            "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            "code_path": str(code_path),
        }
    }
    output = tmp_path / "candidates.json"

    _write_candidates(output, candidates)

    loaded = _load_candidates(output, resume=True)
    assert loaded == candidates
    _validate_existing_candidate(task_uid, loaded[task_uid])
    with pytest.raises(FileExistsError, match="already exists"):
        _load_candidates(output, resume=False)

    code_path.write_text("changed\n")
    with pytest.raises(ValueError, match="SHA256"):
        _validate_existing_candidate(task_uid, loaded[task_uid])
