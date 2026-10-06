# ASPIRE Online-RL service

This package is the no-LLM execution and audit boundary used by the AReaL
`AspireActor`. The public Broker owns episode state, immutable code revisions,
replay quotas, artifacts, and trusted terminal reward. Simulator processes and
optional remote Runner nodes never call a language model.

## Trust boundary

- The Broker accepts only `task_uid` values from a prebuilt campaign manifest.
- Skill and API directories are immutable snapshots; their complete file sets,
  sizes, and SHA256 values are verified at Broker startup and episode creation.
- The Actor prompt, OpenAI tool schema, and model processor have independent
  digests included in the protocol digest and checked by the Actor handshake.
- Generated Python runs in a fresh `multiprocessing.spawn` child for every seed.
  Credentials are removed from the child environment, Molmo is forced off, and
  only approved robot callables and a restricted builtin/import set are exposed.
- `env_reward`, `sandbox_rc`, and `task_completed` come from the runner result
  channel. stdout, stderr, generated variables, and output filenames are never
  trusted as reward inputs.
- This boundary is for accidental-failure isolation, not adversarial Python
  containment. Use the planned container/VM + serialized-API RPC boundary before
  training against actively malicious policies.

Every debug replay uses a fresh process rather than the optional 20-job worker
reuse described in the design. This is a deliberate stricter isolation choice;
it costs simulator startup time but removes cross-seed namespace/state leakage.

## Build an immutable campaign

Generate and review the API documentation that the model may read, then run the
snapshot builder from `ASPIRE/aspire/sim`:

```bash
uv run python scripts/libero/build_online_rl_api_reference.py \
  --output-dir /srv/aspire/approved-api-source
```

After human review, build the immutable snapshots and campaign:

```bash
uv run python scripts/libero/build_online_rl_snapshot.py \
  --skill-source .claude/libero/skills \
  --snapshot-dir /srv/aspire/snapshots/l0-001 \
  --api-source /srv/aspire/approved-api-source \
  --api-snapshot-dir /srv/aspire/snapshots/api-001 \
  --system-prompt ../../../AReaL/examples/aspire/prompts/system.md \
  --findings-prompt ../../../AReaL/examples/aspire/prompts/findings.md \
  --tool-schema ../../../AReaL/examples/aspire/tool_schema.json \
  --model-path "$ASPIRE_ACTOR_MODEL_PATH" \
  --env-config /srv/aspire/configs/libero_goal_swap.yaml \
  --areal-commit "$(git -C ../../../AReaL rev-parse HEAD)" \
  --manifest /srv/aspire/campaigns/goal-swap-v1.json
```

Do not put held-out seeds in this manifest. V1 validates development and
terminal seeds as exactly 51–65.

## Deploy Runner and Broker

Keep tokens in the process environment or a secret manager, never in YAML:

```bash
export ASPIRE_RUNNER_TOKEN='...'
uv run python scripts/libero/serve_online_rl_runner.py \
  --host 0.0.0.0 --port 8126
```

```bash
export ASPIRE_BROKER_ADMIN_TOKEN='...'
export ASPIRE_BROKER_TOKEN_SECRET='...'
export ASPIRE_RUNNER_TOKEN='...'
uv run python scripts/libero/serve_online_rl_broker.py \
  --campaign-manifest /srv/aspire/campaigns/goal-swap-v1.json \
  --state-db /srv/aspire/state/goal-swap-v1.sqlite \
  --artifact-root /srv/aspire/artifacts \
  --api-reference-root /srv/aspire/snapshots/api-001 \
  --runner-url http://runner.internal:8126 \
  --host 0.0.0.0 --port 8125 \
  --max-simulator-concurrency 2
```

SQLite WAL restores active episode state after a Broker restart. If the failed
trainer will not reconnect, restart with
`--cleanup-orphans-older-than-seconds 18000` (or a reviewed, longer threshold)
to mark only stale non-terminal episodes as `INFRA_FAILED`. Cleanup is disabled
by default so a normal Broker restart does not silently destroy resumable work.

Repeat `--runner-url` to distribute jobs round-robin across multiple Runner
nodes. The Broker's global semaphore still caps total simulator concurrency.

Use a private network and TLS/mTLS reverse proxy for cross-host traffic. V1
tokens do not make plaintext HTTP safe on an untrusted network.

Before starting a real simulator workflow, run and inspect these preflight
checks on every Runner host:

```bash
nvidia-smi
curl -fsS http://127.0.0.1:8123/health
curl -fsS http://127.0.0.1:8124/health
uv run python -c 'import libero, mujoco, torch; print(torch.cuda.is_available())'
test -r /srv/aspire/campaigns/goal-swap-v1.json
test -w /srv/aspire/state
test -w /srv/aspire/artifacts
```

Confirm that ports and health paths match the actual SAM3 and Contact-GraspNet
deployment before running commands. A failed simulator/environment initialization
is intentionally an infrastructure error and causes AReaL to discard the group.

## Audit and held-out evaluation

`GET /v1/admin/metrics` reports episode/action/replay counters and invalid-tool
rate. `GET /v1/admin/episodes/{episode_id}/frozen-code` exports the immutable
`fix_code.py` with `X-Code-SHA256`; `GET /v1/admin/episodes/{episode_id}` exports
the state, findings, counters, hashes, and terminal result. These endpoints
require the Broker admin token.
SQLite WAL plus the content-addressed artifact directory are the source of truth.

Held-out evaluation has no gateway or model argument:

```bash
uv run python scripts/libero/eval_frozen_code.py \
  --campaign-manifest /srv/aspire/campaigns/goal-swap-v1.json \
  --task-uid libero_goal_swap/put_the_bowl_on_the_stove \
  --fix-code /srv/aspire/candidates/fix_code.py \
  --output-root /srv/aspire/heldout \
  --resume
```

The evaluator accepts only a sorted subset of seeds 1–50, rechecks the code hash
before every seed and at completion, records `model_calls: 0`, and resumes only
when the immutable identity matches.

## Tests

These tests use a fake runner and do not launch LIBERO:

```bash
uv run --extra dev pytest -q tests/test_online_rl_broker.py
uv run --extra dev ruff check online_rl
```
