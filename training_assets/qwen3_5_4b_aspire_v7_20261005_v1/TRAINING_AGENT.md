# Qwen3.5-4B × ASPIRE v7：AReaL RL 训练资产交接

## 1. 本轮目标与边界

使用 **Qwen3.5-4B 基础权重 + 原封不动的 4B v7 skill**，通过 AReaL v2 的 ASPIRE Actor 工作流进行 LoRA GRPO。v7 是机器人控制知识文本，不是模型 checkpoint，也不是 LoRA adapter。

本轮仅准备资产，**没有启动训练、推理、感知或仿真服务**。也没有实施上一轮讨论的新 skill 更新。下一个 agent 接到明确的训练执行任务后，先完成部署验收，再开始训练。不得用“资产预检通过”替代训练成功。

主机：`Inf_H800_2`。远端项目：`/data/wanghanzhen/projects/EmbodiedAReal`。

资产根目录：

```text
/data/wanghanzhen/projects/EmbodiedAReal/training_assets/qwen3_5_4b_aspire_v7_20261005_v1
├── TRAINING_AGENT.md                 本说明
├── snapshots/skill-v7/               精确复制的冻结 v7
├── snapshots/api/                    v7 配套的冻结 reduced API 文档
├── snapshots/prompts/                当前 Actor system/findings prompts
├── snapshots/tool_schema.json        当前工具 schema
├── snapshots/environment.yaml        冻结环境配置
├── datasets/train_campaign.json      训练 campaign，十任务、开发51–65
├── datasets/task_rows.jsonl          仅十条 task_uid，不含程序或评测反馈
├── configs/acceptance_lora.yaml      一 epoch 的验收训练配置
├── configs/source_lora.yaml          派生前的配置备份
├── configs/resource_plan.json        GPU、CPU、解释器与建议端口
├── models/base_model.json            基础权重外部路径、文件 SHA/大小
├── scripts/env.sh                    路径和非敏感环境变量；不启动进程
├── scripts/secrets.example.sh        必需 secret 名称；没有真实值
├── scripts/verify_assets.py          离线 SHA/源代码/数据/config 解析检查
├── provenance/                      Git HEAD、dirty 状态、源码快照、依赖版本
├── evaluation_reference/            评测结果索引；不得馈入训练
└── state/                           prepared_not_started 与验收 gate 状态
```

## 2. 锁定的初始化资产

- 模型：`/data/wanghanzhen/models/Qwen/Qwen3.5-4B`。不复制大权重，完整文件 SHA 存在 `models/base_model.json`。必须从基础模型开始，不能自动加载历史 run16 的 adapter。
- Skill digest：`f715823eccc1a5e9b2748fd0a1ef7459226b9581196bdcdf85d1b79c6e3a0b33`。
- Skill 来源：`runs/aspire_goal_swap_qwen3_5_4b_codex_skill_loop_20260921_v1/snapshots/skill-v7`。
- Skill 在训练期间固定，不用 rollout findings 或 held-out 结果在线改写。训练改变模型 LoRA 参数。
- API 文档沿用 v7 的 `api-v8`，不要把它与 live API 完全一致视为已验证事实；开始前检查实际 reduced API 的签名和坐标语义。
- 当前 runtime 已冻结源码副本与逐文件 SHA。它与历史 27B runtime 不同；本资产不宣称公平的模型 A/B。
- 两个仓库都有现存未提交修改。Git HEAD 单独不足以复现；还必须通过 `source_hashes.json`，并保留 `source_snapshot/` 和 tracked diff。不要 reset、stash 或覆盖他人的变更。

## 3. 数据与隔离

训练实际读取 `datasets/train_campaign.json`，`load_campaign_rows` 只导出十个 task UID：中层抽屉、碗→炉灶、酒瓶→柜顶、上层抽屉＋碗、碗→柜顶、推盘子到炉灶前、奶酪→碗、开炉灶、碗→盘子、酒瓶→架子。

每个 episode 从冻结 skill/API 和观测生成自己的程序。开发及 terminal seeds 固定 **51–65**。不能把已有冻结候选、成功程序、held-out traces、评测摘要或答案作为 Actor 输入。训练 reward 按协议为 `max(env_reward, 0.1 × normal_execution)`，不得用打印“完成”或程序自述计算成功率。

Held-out **1–50** 是独立外部评测；不配置 `valid_dataset`/`eval_workflow`，不把这些种子塞入训练 campaign。这些种子已在旧实验中使用过，后续成绩应标注为既有 benchmark 的重复评测，不冒称全新未知测试集。

历史 `96/500` 是 stove v3、plate v4、其余 v7 的混合成绩，**不是统一 v7 的 pretraining baseline**。如需测训练收益，应为基础模型和训练模型使用本资产的同一 v7、runtime、开发预算生成各自候选，冻结后再外部评测；旧结果仅用于溯源。

## 4. 训练语义与资源

当前支持且准备的是 **LoRA 验收配置**，不是 full-parameter 训练。

- GPU0–1：FSDP Actor DP2；GPU2–5：SGLang TP4；GPU6：SAM3；GPU7：Contact-GraspNet及仿真 EGL。PyRoKi 使用 CPU。
- AReaL 只暴露 GPU0–5，`cluster.n_gpus_per_node=6`。不要把感知服务进程放入训练 collective。
- 每任务 K=4，同任务组内 population mean/std 做 reward normalization；每 batch 两个 task group，即 **8 个 episode/update**。
- 十任务、一 epoch 共 **40 个 episode、5 个训练 step**。这是小规模验收，不代表足够训练。`AspireGRPOConfig` 当前强制一 epoch，不能直接改成多 epoch；扩大训练方案需单独修改和验证。
- `episode_mean`：每 episode 总 loss mass 相同。HF `individual` export 保留每 turn 完整历史；不能直接改 concat、摘要、截断或 token_mean。
- `offpolicy=0`；温度1；每 turn 输出4096；episode输出预算65536；上下文131072；无 ref/critic/teacher/KL；不做第二次 reward/adv normalization。
- LoRA rank16/alpha16，q/k/v/o，预期32个 target module；BF16、gradient checkpointing、selective logits。
- 推理并发4、static memory fraction0.60、CUDA graph关闭。TP2长上下文 headroom 不足的历史风险不能忽略。
- Actor 必须使用 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- Broker 建议先用3个独立 runner、`max-simulator-concurrency=3`；这是初始资源规划，未在训练 rollout 压测。runner、感知服务均须有独立日志与健康检查。
- Native CPU threads 每 runner 4，防止多个 spawn 进程各开启全机线程。启动前重新检查 GPU、PID、端口；资产准备时的空闲状态不是预留。

## 5. 已知阻塞：必须先验证

历史 run16g 在 PPO 更新返回 `/ppo/actor/update` 502；更底层有 rank0 `ALLGATHER Seq17430` 等待7200秒超时，而 rank1最后 enqueue17429。**FSDP rank 分支与 collective 顺序一致性没有得到本资产的运行验证**。不要仅延长超时来掩盖问题。

历史证据路径：

```text
/data/wanghanzhen/projects/EmbodiedAReal/runs/aspire_goal_swap_qwen3_5_4b_original_skill_grpo_20260916_v3_sha_managed/training/run16/logs/lora-v1-sha-run16g-tp4-c4-m60-expandable-direct-all.log
```

当前 actor 已实现全 DP 的零优势跳过 optimizer，但需真实回归 rank 一致性。零方差组或 adapter 文件存在，均不能证明模型学习发生。必须记录 group rewards、skip标记、有限 loss/grad、optimizer step、参数变化、adapter SHA 与推理 version。

另有历史 context overflow、worker readiness 超时、`/save` 502、warmup失败，以及 cleanup中的 `PPOTrainer.saver` AttributeError。先查最初错误，不要只查最后一条异常。

本次没有执行或找到已通过的独立 **131k VLM forward/backward canary**。说明中的版本 metadata 也不等于 CUDA 扩展可加载。

## 6. 下一个 agent 的执行顺序

1. 读两个仓库 AGENTS 和 ASPIRE LIBERO 协议；确认收到训练执行授权。
2. 检查 `state/preparation_status.json`、运行目录、PID/GPU/端口；选择新的独立 run，不覆盖旧实验。
3. 离线验证、再运行 focused tests；不启动 Trainer：

```bash
cd /data/wanghanzhen/projects/EmbodiedAReal/AReaL
source /data/wanghanzhen/projects/EmbodiedAReal/training_assets/qwen3_5_4b_aspire_v7_20261005_v1/scripts/env.sh
.venv/bin/python "$ASPIRE_TRAIN_ASSET/scripts/verify_assets.py" --asset-root "$ASPIRE_TRAIN_ASSET" --full-model
.venv/bin/python -m pytest -q tests/test_aspire_config.py tests/test_aspire_agent.py tests/test_aspire_generate_candidates.py tests/test_episode_loss_reduction.py
```

4. 在私有文件提供 fresh per-run secrets；不要复用或打印历史 token。创建 `$ASPIRE_TRAIN_RUN/{logs,state,broker,artifacts,experiments,name-resolve,canary}`。name-resolve、Broker DB、checkpoint目录全部隔离。
5. 用 LIBERO `.venv-libero/bin/python3` 启动 SAM3/GraspNet/PyRoKi、独立 runners、Broker。Broker manifest 指向本资产；API root 指向 `snapshots/api`；状态与 artifacts 指向新 run。建议 Broker8625、runner8626–8628、感知8114–8116，先检查端口。
6. 使用 AReaL `.venv/bin/python` 做正式 preflight（含 fast-path 符号 import、prompt/schema/processor、chat template、Broker）；网关启动后再做 `--probe-gateway`，其模型调用必须 `store=false`。

```bash
.venv/bin/python -m examples.aspire.preflight --campaign-manifest "$ASPIRE_CAMPAIGN_MANIFEST" --model-path "$ASPIRE_ACTOR_MODEL_PATH" --broker-url "$ASPIRE_BROKER_URL"
```

7. 先过真实部署 canary：完整131072-token VLM forward/backward、四并发 rollout/parser/LoRA、两 FSDP rank 的非零优势 update、零优势分支一致性、首次 disk adapter 发布及推理版本同步。没有现成已验收 canary 脚本，训练 agent 应编写或定位适用入口；不能把下面完整 campaign 当成预检。
8. canary通过后才运行一epoch验收训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv/bin/python -m examples.aspire.train --config "$ASPIRE_TRAIN_ASSET/configs/acceptance_lora.yaml"
```

9. 每个 gate 把时间、命令、源码 SHA、结果与日志路径写入新 run。一个 step 的流程完成不等于 optimizer真实更新；记录 skip 和 version。失败保留第一现场，停止该 run 自己的进程；不要全机 `pkill`。当前 recover disabled，不保证跨步恢复；重新启动需新 trial 或经过验证的恢复路径。
10. 训练后为明确选定的 checkpoint重新生成十任务开发候选；核验冻结 SHA；停止或隔离模型；外部 evaluator运行每任务50个held-out seed，模型调用为零。零分/异常同样应保留，不能看 held-out 后反复改代码仍叫同一 frozen 评测。

## 7. 本次已完成与尚未完成

已完成：原始v7复制和digest验证、配套API与环境快照、十任务数据隔离、模型文件身份记录、当前源码与dirty修改留档、独立资源/路径规划、LoRA配置派生与离线检查入口。

实际离线检查结果见 `state/offline_verification.json`，不是本说明中的计划文字。所有在线 readiness、parser probe、GPU canary、训练更新和post-training评测默认 **pending**。未选择大规模训练时长或新增优化；下一个 agent 不应自行宣称已有训练收益。
