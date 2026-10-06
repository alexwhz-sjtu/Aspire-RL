# Source this file from a trusted shell. It does not start services.
export ASPIRE_TRAIN_ASSET=/data/wanghanzhen/projects/EmbodiedAReal/training_assets/qwen3_5_4b_aspire_v7_20261005_v1
export ASPIRE_ACTOR_MODEL_PATH=/data/wanghanzhen/models/Qwen/Qwen3.5-4B
export ASPIRE_CAMPAIGN_MANIFEST="$ASPIRE_TRAIN_ASSET/datasets/train_campaign.json"
export ASPIRE_TRAIN_RUN="${ASPIRE_TRAIN_RUN:-/data/wanghanzhen/projects/EmbodiedAReal/runs/aspire_qwen3_5_4b_v7_rl_20261005_acceptance_v1}"
export AREAL_EXPERIMENT_ROOT="$ASPIRE_TRAIN_RUN/experiments"
export AREAL_NAME_RESOLVE_ROOT="$ASPIRE_TRAIN_RUN/name-resolve"
export ASPIRE_BROKER_URL="${ASPIRE_BROKER_URL:-http://127.0.0.1:8625}"
export ASPIRE_ROOT=/data/wanghanzhen/projects/EmbodiedAReal/ASPIRE/aspire/sim
export PYTHON_ROOT=/data/wanghanzhen/projects/EmbodiedAReal/ASPIRE
export PYTHONPATH=/data/wanghanzhen/projects/EmbodiedAReal/AReaL:/data/wanghanzhen/projects/EmbodiedAReal/ASPIRE
export MUJOCO_GL=egl
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export ASPIRE_DISABLE_MOLMO=1
export HF_HUB_OFFLINE=1
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export ASPIRE_SIMULATOR_THREADS=4
# Native simulator libraries are managed by the existing LIBERO environment.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy LD_LIBRARY_PATH
