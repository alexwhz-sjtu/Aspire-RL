# ASPIRE Actor online GRPO example

`AspireActor` is a plain, reentrant AReaL v2 agent. One `run()` owns one Broker
episode and returns one terminal `float` reward. All state is local to `run()`;
the shared Agent instance contains only immutable configuration and a read-only
processor/tokenizer.

The training trajectory contains every `store=true` assistant completion in the
coding/debugging loop. Reasoning, tool names/arguments, code, patches,
`final_submission`, and EOS are trainable. System/user/tool/image/padding tokens
are masked. The post-freeze findings request is `store=false`, has no tools, is
best-effort, and cannot enter the trajectory or reward.

The model-facing tools do not expose code hashes. The episode-local runtime
records the SHA256 returned by each successful `write_edit` and injects it into
subsequent edit, replay, and final-submission Broker requests. The Broker still
performs the authoritative hash and optimistic-state validation.

## AReaL integration choices

- `rollout.agent.mode: inline`; AReaL's external-session `online` mode is not used.
- `export_style: individual` and `chat_template_type: hf` preserve each policy
  completion while masking its complete repeated history.
- `gconfig.n_samples: 4` creates four independent sessions per task row.
- rollout-time population mean/std normalization broadcasts one normalized
  terminal return to all policy calls from that session. Equal group rewards are
  logged as `zero_variance_group=true`; all-zero advantages skip the optimizer
  step so AdamW weight decay cannot create a fake policy update.
- `actor.loss_reduction: episode_mean` attaches `loss_mask / N_episode` before
  session boundaries are merged. FSDP/Megatron's existing global loss-weight
  aggregation then gives every episode total mass one, independent of its number
  of calls or action tokens. The default `token_mean` behavior is unchanged.
- Policy failures are structured Broker results and normally converge to a valid
  reward. Auth/storage/service/environment corruption remains an infrastructure
  exception, for which AReaL discards the whole group.

The Aspire-specific config rejects SAPO, CISPO, teacher distillation, decoupled
loss, non-individual export, reward/advantage renormalization, KL shaping, and
non-v2 controllers when `episode_mean` is active.

## Preflight

Set paths and secrets without committing them:

```bash
export ASPIRE_ACTOR_MODEL_PATH=/share/dai-sys/wanghanzhen/models/Qwen/Qwen3.5-4B
export ASPIRE_CAMPAIGN_MANIFEST=/srv/aspire/campaigns/goal-swap-v1.json
export ASPIRE_BROKER_URL=http://broker.internal:8125
export ASPIRE_BROKER_ADMIN_TOKEN='...'
export AREAL_ADMIN_API_KEY='...'
export AREAL_EXPERIMENT_ROOT=/srv/areal/experiments
export AREAL_NAME_RESOLVE_ROOT=/srv/areal/name-resolve
```

Run the read-only digest, processor/chat-template, and Broker readiness gate:

```bash
uv run python -m examples.aspire.preflight \
  --campaign-manifest "$ASPIRE_CAMPAIGN_MANIFEST" \
  --model-path "$ASPIRE_ACTOR_MODEL_PATH" \
  --broker-url "$ASPIRE_BROKER_URL"
```

For Qwen3.5 this gate also requires the environment-compatible FlashAttention,
flash-linear-attention (`fla`), and `causal_conv1d` fast paths. The 131k profile
is rejected when they are absent; do not rely on Transformers' slow fallback.

After the AReaL inference gateway is ready, exercise one real parser request
(this is a model call but is `store=false`):

```bash
export AREAL_GATEWAY_API_KEY='...'
uv run python -m examples.aspire.preflight \
  --campaign-manifest "$ASPIRE_CAMPAIGN_MANIFEST" \
  --model-path "$ASPIRE_ACTOR_MODEL_PATH" \
  --broker-url "$ASPIRE_BROKER_URL" \
  --gateway-url http://127.0.0.1:8080 \
  --probe-gateway
```

The 131072-token VLM forward/backward gate is deliberately not run by preflight:
it needs the final 8×H800 allocation. Run it as a deployment canary before the
first campaign and do not silently lower the context limit on failure.

## Train

Expose only GPUs 0–5 to AReaL. GPUs 6–7 remain in separate perception-service
processes and are not part of `cluster.n_gpus_per_node`.
The LoRA profile assigns GPUs 0–1 to the two FSDP Actor ranks and GPUs 2–5 to
the SGLang TP=4 rollout worker. TP=4 and a 0.60 static-memory fraction leave
activation headroom for four concurrent 131072-token Qwen3.5 trajectories.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 uv run python -m examples.aspire.train \
  --config examples/aspire/configs/qwen3_5_4b_lora.yaml
```

The LoRA profile has two task groups per update and K=4, so each optimizer step
contains eight real Actor episodes. The two outer trajectories are divided across
the two FSDP ranks, while reward normalization remains independent inside each
four-session task group. Ten Goal-Swap rows and one epoch produce five optimizer
steps and forty episodes. Both actor and rollout controllers are explicitly v2.
LoRA uses AReaL's
automatic disk adapter synchronization; the full-parameter dry profile selects
AWEX automatically and does not configure legacy XCCL. `train.py` rejects the
full-parameter profile so it cannot accidentally launch the complete campaign;
use that profile only in the deployment-specific dry-forward/backward gate.

## Candidate generation and held-out handoff

Generate each checkpoint's own development candidate and export immutable code:

```bash
uv run python -m examples.aspire.generate_candidates \
  --broker-url "$ASPIRE_BROKER_URL" \
  --gateway-url http://127.0.0.1:8080 \
  --model-path "$ASPIRE_ACTOR_MODEL_PATH" \
  --task-uid libero_goal_swap/put_the_bowl_on_the_stove \
  --output /srv/aspire/candidates/trained.json
```

Pass the exported code to ASPIRE's separate `eval_frozen_code.py`. Do not use
`PPOTrainer.valid_dataset` or `eval_workflow`: held-out seeds 1–50 must execute
the same code hash with zero gateway calls.

## Focused tests

```bash
uv run pytest -q tests/test_aspire_agent.py
uv run pytest -q tests/test_episode_loss_reduction.py tests/test_functional.py
uv run pytest -q tests/v2/inference_service/test_data_proxy_chat.py
```
