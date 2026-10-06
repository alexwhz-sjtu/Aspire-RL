# Aspire-RL

Qwen3.5-4B 机器人程序策略的 AReaL v2 / ASPIRE 在线 LoRA GRPO 框架。

## 分支

- `v1`：2026-10-06 从 `Inf_H800_2` 导出的当前代码基线，包括原工作目录的未提交代码修改。
- `v2`：从同一基线创建，供后续吞吐、仿真、感知和训练流水线优化。

这是源码版本快照，不代表训练、131k VLM canary 或 held-out 评测已经通过。
本次发布没有改变原服务器工作目录，也没有启动训练或服务。

## 目录

- `AReaL/`：训练与推理基础设施、`examples/aspire/` Actor 和相关测试。
- `ASPIRE/`：Broker、隔离 runner、LIBERO、感知/运动 API 与相关测试。
- `training_assets/qwen3_5_4b_aspire_v7_20261005_v1/`：冻结 v7 skill/API、训练配置、campaign 和部署交接材料。
- `patches/`：当前第三方依赖的本地修改。
- `provenance/current_snapshot.json`：上游 commit、捕获时的 dirty 状态、导出文件 SHA256、依赖版本及排除项。
- `scripts/bootstrap_submodules.py`：检出固定版本的依赖并应用记录的补丁。
- `scripts/verify_snapshot.py`：检查发布源码是否与捕获内容一致。

训练资产是 2026-10-05 的部署参考，当前代码快照是 2026-10-06。
两者的源代码身份可能不同。资产中的绝对路径、模型 SHA 和验收状态均保留原值，
不要把旧的离线检查报告当作新 checkout 的部署验收。
历史评测索引及完整旧源码重复副本未包含在训练资产发布材料中。
在新路径部署时，应重新生成 campaign/资产身份并执行交接说明要求的 gate。

## 克隆与依赖

```bash
git clone --branch v2 https://github.com/alexwhz-sjtu/Aspire-RL.git
cd Aspire-RL
python3 scripts/verify_snapshot.py
python3 scripts/bootstrap_submodules.py
```

默认初始化 LIBERO、两个 robosuite 依赖、SAM3 和 Contact-GraspNet。
需要 BEHAVIOR/cuRobo 时使用 `python3 scripts/bootstrap_submodules.py --all`。
这些操作仅准备依赖源码，不安装环境、不下载权重、不启动服务。

Contact-GraspNet 有三处与原服务器一致的兼容修改；bootstrap 会应用
`patches/contact_graspnet_pytorch.patch` 并验证修改文件 SHA。
直接执行 `git submodule update --init --recursive` 不会自动应用该补丁。

依赖安装请分别遵循 `AReaL/README.md`、`ASPIRE/aspire/sim/README.md`，
使用相互独立的 Python 环境。模型、感知权重、CUDA 扩展和本地动态库均需另行准备。
上游及第三方的许可证保留在对应项目中。

## 在线训练入口

1. 阅读 `training_assets/qwen3_5_4b_aspire_v7_20261005_v1/TRAINING_AGENT.md`。
2. 在当前 checkout 下核验 reduced API、冻结 skill 和 campaign 身份。
3. 准备独立 run、端口、GPU 映射及外部 secret 文件。
4. 启动 SAM3、GraspNet、CPU PyRoKi、runner 和 Broker，执行 readiness/preflight。
5. 完成真实 131k VLM forward/backward、双 rank 优势分支和 LoRA/version canary。
6. 再运行 `AReaL/examples/aspire/train.py` 的 LoRA 验收配置。

原资产的 `scripts/env.sh` 指向原服务器目录。切换源码时，必须核对并设置
`PYTHONPATH`、`ASPIRE_ROOT` 和 `PYTHON_ROOT`，避免在 `v2` 工作目录下仍执行旧代码。

固定语义：每任务 K=4、两个 task group/update、`hf + individual` 导出、
`episode_mean`、offpolicy=0；开发及 terminal seeds 51–65，外部 held-out 1–50。
冻结 skill 不随训练改写。thinking 默认由 Qwen3.5 的模型模板控制。
仅修改分支名称不会改变这些训练行为。

## 发布范围

导出了两个仓库的当前受版本管理文件、允许的新增测试和 skill 文件。
第三方仓库以固定 Git submodule commit 保留，本地修改以可验证补丁保留。
不包含模型/adapter 权重、运行日志、Broker 数据库、实验回放、held-out 答案、
虚拟环境、机器上的动态库或真实凭据。

`v2` 优化应逐项测量 queue、prefill/decode、render/readback、感知服务锁、
终评、DP 负载和 FSDP collective。减少 seed、改 reward/上下文、开启旧策略异步
或改变轨迹拼接语义需要单独验证，不属于本次发布。
