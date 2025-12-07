#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/../.."

# 支持通过参数或环境变量传递配置
DATASET="${DATASET:-${1:-Industrial_and_Scientific}}"
ROOT="${ROOT:-${2:-${PROJECT_ROOT}/data/Amazon18/${DATASET}}}"
PLM_CHECKPOINT="${PLM_CHECKPOINT:-${3:-your_emb_model_path}}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

echo "Converting text to embeddings..."
echo "Dataset: $DATASET"
echo "Data root: $ROOT"
echo "PLM checkpoint: $PLM_CHECKPOINT"
echo "Number of processes: $NUM_PROCESSES"

# 验证数据目录存在
if [ ! -d "$ROOT" ]; then
    echo "Error: Data directory does not exist: $ROOT"
    exit 1
fi

accelerate launch --num_processes "$NUM_PROCESSES" "${SCRIPT_DIR}/amazon_text2emb.py" \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --plm_checkpoint "$PLM_CHECKPOINT"
