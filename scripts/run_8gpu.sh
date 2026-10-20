#!/usr/bin/env bash
set -e

# 自动定位项目根目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"

echo "================================================================================="
echo ">>> [Project BUILD 3.0] 启动 8 卡 NVIDIA GPU 分布式并行训练 (torchrun DDP)"
echo "================================================================================="

# 动态随机选取未被占用的 Master 通信端口，杜绝端口冲突
MASTER_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
export MASTER_ADDR="127.0.0.1"
export MASTER_PORT=${MASTER_PORT}
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=4

echo ">>> NCCL 主通信端口绑定: ${MASTER_PORT}"
echo ">>> 启动 8 进程并行计算加速..."

torchrun \
    --nproc_per_node=8 \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_target_sbdd_ddp.py \
    "$@"
