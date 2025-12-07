#!/bin/bash
# Bad Case 分析脚本

set -e

echo "=================================================="
echo "  Bad Case 分析工具"
echo "=================================================="

# 默认参数
CATEGORY="Industrial_and_Scientific"
MODEL_PATH=""
TOP_K=10

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --category)
            CATEGORY="$2"
            shift 2
            ;;
        --top_k)
            TOP_K="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# 检查必要参数
if [ -z "${MODEL_PATH}" ]; then
    echo "错误: 必须指定模型路径 --model"
    echo "用法: bash analyze_badcase.sh --model /path/to/model --category Industrial_and_Scientific"
    exit 1
fi

# 数据路径
TEST_DATA="./data/Amazon/test/${CATEGORY}*11.csv"
ITEM_META="./data/Amazon/index/${CATEGORY}.item.json"
SID_INDEX="./data/Amazon/index/${CATEGORY}.index.json"

# 评估结果路径
EVAL_OUTPUT="./analysis_output/${CATEGORY}_$(date +%Y%m%d_%H%M)"
mkdir -p ${EVAL_OUTPUT}

echo "模型路径: ${MODEL_PATH}"
echo "数据集: ${CATEGORY}"
echo "Top-K: ${TOP_K}"
echo "输出目录: ${EVAL_OUTPUT}"
echo ""

# ==================== 步骤 1: 运行评估生成预测结果 ====================
echo "【步骤 1/3】 运行评估生成预测结果"
echo "--------------------------------------------------"

EVAL_RESULT_FILE="${EVAL_OUTPUT}/eval_results.jsonl"

python evaluate.py \
    --base_model ${MODEL_PATH} \
    --test_file ${TEST_DATA} \
    --batch_size 8 \
    --num_beams 50 \
    --max_new_tokens 256 \
    --output_dir ${EVAL_OUTPUT} \
    --category ${CATEGORY} \
    --sid_index_path ${SID_INDEX} \
    --item_meta_path ${ITEM_META} \
    --save_predictions True \
    --prediction_file ${EVAL_RESULT_FILE}

echo "评估完成,结果保存在: ${EVAL_RESULT_FILE}"

# ==================== 步骤 2: 分析 Bad Cases ====================
echo ""
echo "【步骤 2/3】 分析 Bad Cases"
echo "--------------------------------------------------"

python analyze_badcase.py \
    --eval_results ${EVAL_RESULT_FILE} \
    --test_data ${TEST_DATA} \
    --item_meta ${ITEM_META} \
    --sid_index ${SID_INDEX} \
    --output_dir ${EVAL_OUTPUT} \
    --top_k ${TOP_K}

# ==================== 步骤 3: 生成可视化报告 ====================
echo ""
echo "【步骤 3/3】 生成可视化报告"
echo "--------------------------------------------------"

python visualize_analysis.py \
    --report_path ${EVAL_OUTPUT}/badcase_report.json \
    --output_dir ${EVAL_OUTPUT}/visualizations

echo ""
echo "=================================================="
echo "  分析完成!"
echo "=================================================="
echo "分析报告: ${EVAL_OUTPUT}/badcase_report.json"
echo "可视化结果: ${EVAL_OUTPUT}/visualizations/"
echo ""
echo "下一步:"
echo "1. 查看分析报告,了解主要失败模式"
echo "2. 根据优化建议调整模型或训练策略"
echo "3. 重新训练并对比效果"
echo "=================================================="
