"""Trusted CAP-RL reward calculation."""

from __future__ import annotations

import math
from collections.abc import Iterable


def cap_rl_seed_reward(env_reward: float, sandbox_rc: int) -> float:
    """Return the CAP-RL score for one trusted simulator result."""
    reward = float(env_reward)
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise ValueError(f"env_reward must be finite and in [0, 1], got {env_reward!r}")
    execution_bonus = 0.1 if int(sandbox_rc) == 0 else 0.0
    return max(reward, execution_bonus)


def cap_rl_episode_reward(results: Iterable[tuple[float, int]]) -> float:
    """Average CAP-RL scores; reject empty or malformed terminal evaluations."""
    scores = [cap_rl_seed_reward(reward, rc) for reward, rc in results]
    if not scores:
        raise ValueError("terminal evaluation must contain at least one seed")
    return sum(scores) / len(scores)
