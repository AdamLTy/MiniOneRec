#!/bin/bash
# MiniOneRec 0.5B 快速启动脚本
# 用于快速验证流程 (使用小数据集和简化配置)

set -e

echo "=================================================="
echo "  MiniOneRec 0.5B 快速启动"
echo "  用于快速验证流程,非完整训练"
echo "=================================================="

# 配置
CATEGORY="Industrial_and_Scientific"
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
SAMPLE_SIZE=5000  # 仅使用 5000 样本快速验证

export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1

echo "⚠️  快速验证模式:"
echo "  - 仅使用 ${SAMPLE_SIZE} 训练样本"
echo "  - 训练 1 个 epoch"
echo "  - 跳过 RL 阶段"
echo ""
read -p "继续? [Y/n]: " confirm
confirm=${confirm:-Y}

if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "已取消"
    exit 0
fi

# 检查数据是否已准备
if [ ! -f "./data/Amazon/train/${CATEGORY}*11.csv" ]; then
    echo "错误: 数据未准备,请先运行完整 pipeline 或手动准备数据"
    echo "提示: bash pipeline_0.5b.sh"
    exit 1
fi

# 检查 SID 索引
if [ ! -f "./data/Amazon/index/${CATEGORY}.index.json" ]; then
    echo "错误: SID 索引未生成,请先构建 SID"
    exit 1
fi

# 创建输出目录
OUTPUT_DIR="./outputs/quickstart_$(date +%Y%m%d_%H%M)"
mkdir -p ${OUTPUT_DIR}

echo ""
echo "【步骤 1/3】 快速 SFT 训练 (1 epoch, ${SAMPLE_SIZE} 样本)"
echo "--------------------------------------------------"

python sft.py \
    --base_model ${BASE_MODEL} \
    --batch_size 64 \
    --micro_batch_size 16 \
    --num_epochs 1 \
    --learning_rate 5e-4 \
    --cutoff_len 256 \
    --train_file ./data/Amazon/train/${CATEGORY}*11.csv \
    --eval_file ./data/Amazon/valid/${CATEGORY}*11.csv \
    --output_dir ${OUTPUT_DIR}/sft_model \
    --wandb_project "MiniOneRec-Quickstart" \
    --wandb_run_name "quickstart-sft-$(date +%m%d_%H%M)" \
    --use_swanlab True \
    --category ${CATEGORY} \
    --train_from_scratch False \
    --freeze_LLM False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/${CATEGORY}.index.json \
    --item_meta_path ./data/Amazon/index/${CATEGORY}.item.json \
    --sample ${SAMPLE_SIZE}

echo ""
echo "【步骤 2/3】 快速评估"
echo "--------------------------------------------------"

python evaluate.py \
    --base_model ${OUTPUT_DIR}/sft_model \
    --test_file ./data/Amazon/test/${CATEGORY}*11.csv \
    --batch_size 16 \
    --num_beams 20 \
    --max_new_tokens 256 \
    --output_dir ${OUTPUT_DIR}/evaluation \
    --category ${CATEGORY} \
    --sid_index_path ./data/Amazon/index/${CATEGORY}.index.json \
    --item_meta_path ./data/Amazon/index/${CATEGORY}.item.json \
    --save_predictions True \
    --prediction_file ${OUTPUT_DIR}/evaluation/eval_results.jsonl

echo ""
echo "【步骤 3/3】 Bad Case 分析"
echo "--------------------------------------------------"

python analyze_badcase.py \
    --eval_results ${OUTPUT_DIR}/evaluation/eval_results.jsonl \
    --test_data ./data/Amazon/test/${CATEGORY}*11.csv \
    --item_meta ./data/Amazon/index/${CATEGORY}.item.json \
    --sid_index ./data/Amazon/index/${CATEGORY}.index.json \
    --output_dir ${OUTPUT_DIR}/analysis \
    --top_k 10

python visualize_analysis.py \
    --report_path ${OUTPUT_DIR}/analysis/badcase_report.json \
    --output_dir ${OUTPUT_DIR}/analysis/visualizations

echo ""
echo "=================================================="
echo "  快速验证完成!"
echo "=================================================="
echo "输出目录: ${OUTPUT_DIR}"
echo ""
echo "查看结果:"
echo "  1. 评估指标: cat ${OUTPUT_DIR}/evaluation/results.json"
echo "  2. Bad Case 报告: cat ${OUTPUT_DIR}/analysis/badcase_report.json"
echo "  3. 可视化: 查看 ${OUTPUT_DIR}/analysis/visualizations/"
echo ""
echo "下一步:"
echo "  - 运行完整训练: bash pipeline_0.5b.sh"
echo "  - 应用优化策略: python optimization_strategies.py --badcase_report ${OUTPUT_DIR}/analysis/badcase_report.json"
echo "=================================================="
