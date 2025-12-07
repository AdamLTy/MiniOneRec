#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# 支持通过参数或环境变量传递配置
DATASET_NAME="${1:-${DATASET_NAME:-Industrial_and_Scientific}}"
INPUT_DIR="${2:-${DATA_DIR:-${PROJECT_ROOT}/data/Amazon18/${DATASET_NAME}}}"
OUTPUT_DIR="${3:-${OUTPUT_DIR:-${PROJECT_ROOT}/data/Amazon}}"

PYTHON_SCRIPT="${PROJECT_ROOT}/convert_dataset.py"

# ===========================================

echo "Start converting $DATASET_NAME ..."
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# 验证输入目录存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

python "$PYTHON_SCRIPT" \
    --dataset_name "$DATASET_NAME" \
    --data_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --category "$DATASET_NAME" \
    --seed 42

echo "Finished!"
