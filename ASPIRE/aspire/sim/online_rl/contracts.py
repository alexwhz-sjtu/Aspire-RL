"""Versioned wire contracts shared by the ASPIRE broker and AReaL actor."""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

PROTOCOL_VERSION = "aspire-online-rl/v1"
DEVELOPMENT_SEEDS = tuple(range(51, 66))
HELDOUT_SEEDS = tuple(range(1, 51))


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpisodeState(str, Enum):  # noqa: UP042 - ASPIRE supports Python 3.10
    CREATED = "created"
    CONTEXT_READY = "context_ready"
    INITIAL_CODE_WRITTEN = "initial_code_written"
    DEBUGGING = "debugging"
    CODE_FROZEN = "code_frozen"
    REPORTING = "reporting"
    TERMINAL_SCORING = "terminal_scoring"
    COMPLETED = "completed"
    POLICY_FAILED = "policy_failed"
    INFRA_FAILED = "infra_failed"
    CLOSED = "closed"


class ErrorClass(str, Enum):  # noqa: UP042 - ASPIRE supports Python 3.10
    POLICY = "policy"
    INFRASTRUCTURE = "infrastructure"


class PolicyError(StrictModel):
    error_class: Literal[ErrorClass.POLICY] = ErrorClass.POLICY
    code: str
    message: str
    retryable: bool = False


class ArtifactRef(StrictModel):
    artifact_id: str
    sha256: str
    media_type: str
    size_bytes: int = Field(ge=0)
    kind: str


class Observation(StrictModel):
    task_language: str
    text: str
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class EpisodeBudgets(StrictModel):
    max_model_turns: int = Field(default=48, ge=1)
    max_output_tokens_per_turn: int = Field(default=4096, ge=1)
    max_policy_output_tokens: int = Field(default=65536, ge=1)
    max_repair_replays_per_seed: int = Field(default=3, ge=0)
    replay_timeout_seconds: int = Field(default=600, ge=1)
    episode_timeout_seconds: int = Field(default=14400, ge=1)


class CampaignTask(StrictModel):
    task_uid: str
    suite: str
    task_name: str
    task_id: int = Field(ge=0)
    task_language: str
    env_config: str
    env_config_digest: str
    skill_snapshot_id: str
    skill_snapshot: str
    skill_snapshot_digest: str
    protocol_digest: str
    development_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    terminal_seeds: tuple[int, ...] = DEVELOPMENT_SEEDS
    budgets: EpisodeBudgets = Field(default_factory=EpisodeBudgets)

    @field_validator("development_seeds", "terminal_seeds")
    @classmethod
    def validate_development_seeds(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if tuple(sorted(set(value))) != DEVELOPMENT_SEEDS:
            raise ValueError("V1 development and terminal seeds must be exactly 51..65")
        return value


class CampaignManifest(StrictModel):
    schema_version: int = 1
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    campaign_id: str
    protocol_digest: str
    api_snapshot: str
    api_snapshot_digest: str
    actor_prompt_digest: str
    tool_schema_digest: str
    model_path: str
    processor_digest: str
    aspire_commit: str | None = None
    areal_commit: str | None = None
    tasks: list[CampaignTask] = Field(min_length=1)


class CreateEpisodeRequest(StrictModel):
    protocol_version: Literal[PROTOCOL_VERSION] = PROTOCOL_VERSION
    task_uid: str
    client_episode_id: UUID


class EpisodeCreated(StrictModel):
    episode_id: UUID
    scoped_token: str
    state: EpisodeState
    state_version: int
    initial_observation: Observation
    skill_snapshot_digest: str
    api_snapshot_digest: str
    actor_prompt_digest: str
    tool_schema_digest: str
    processor_digest: str
    protocol_digest: str
    budgets: EpisodeBudgets


class ReadSearchAction(StrictModel):
    type: Literal["read_search"]
    root: Literal["workspace", "skills", "api_reference"]
    path: str | None = Field(default=None, max_length=4096)
    pattern: str | None = Field(default=None, max_length=1024)
    max_bytes: int = Field(default=32768, ge=1, le=131072)


class WriteEditAction(StrictModel):
    type: Literal["write_edit"]
    mode: Literal["replace", "unified_patch"] = "replace"
    content: str = Field(max_length=262144)
    base_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ReplayAction(StrictModel):
    type: Literal["replay"]
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    seeds: list[int] = Field(min_length=1, max_length=15)

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("replay seeds must be unique")
        invalid = sorted(set(value).difference(DEVELOPMENT_SEEDS))
        if invalid:
            raise ValueError(f"replay seeds must be in 51..65: {invalid}")
        return value


class TraceInspectionAction(StrictModel):
    type: Literal["trace_inspection"]
    replay_ids: list[UUID] = Field(min_length=1, max_length=8)
    selectors: list[
        Literal["trace", "stdout", "stderr", "rgb_keyframes", "mask", "grasp", "ik"]
    ] = Field(min_length=1, max_length=7)
    max_bytes: int = Field(default=32768, ge=1, le=131072)
    max_images: int = Field(default=2, ge=0, le=2)


class FinalSubmissionAction(StrictModel):
    type: Literal["final_submission"]
    code_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


EpisodeAction = Annotated[
    ReadSearchAction
    | WriteEditAction
    | ReplayAction
    | TraceInspectionAction
    | FinalSubmissionAction,
    Field(discriminator="type"),
]


class ActionRequest(StrictModel):
    request_id: UUID
    expected_state_version: int = Field(ge=0)
    model_turn: int | None = Field(default=None, ge=1, le=48)
    cumulative_policy_output_tokens: int | None = Field(
        default=None, ge=0, le=65536
    )
    action: EpisodeAction


class EpisodeCounters(StrictModel):
    code_revisions: int = 0
    baseline_replays: dict[int, int] = Field(default_factory=dict)
    repair_replays: dict[int, int] = Field(default_factory=dict)


class ActionResponse(StrictModel):
    ok: bool
    state: EpisodeState
    state_version: int
    result: dict[str, Any] | None = None
    error: PolicyError | None = None
    counters: EpisodeCounters


class SeedReplayResult(StrictModel):
    replay_id: UUID
    seed: int
    code_sha256: str
    env_reward: float = Field(ge=0.0, le=1.0)
    sandbox_rc: int
    task_completed: bool
    trace_summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)


class FindingsRequest(StrictModel):
    findings: str = Field(max_length=131072)


class TerminalScoreRequest(StrictModel):
    auto_freeze: bool = False
    fallback_findings: str | None = Field(default=None, max_length=8192)


class TerminalScoreResponse(StrictModel):
    code_sha256: str
    reward: float = Field(ge=0.0, le=1.0)
    seeds: list[SeedReplayResult]


class CloseEpisodeRequest(StrictModel):
    final_state: EpisodeState | None = None


class HealthResponse(StrictModel):
    status: Literal["ok", "not_ready"]
    protocol_version: str
    detail: str | None = None
