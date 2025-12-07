#!/bin/bash
# 单卡 4090 24G + SwanLab 训练脚本 - 使用 0.5B 模型
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0  # 指定使用第 0 号 GPU

# 推荐使用的 0.5B 模型:
# - Qwen/Qwen2.5-0.5B-Instruct (推荐)
# - meta-llama/Llama-3.2-1B-Instruct
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"  # 修改为你的本地模型路径

# 数据集设置
CATEGORY="Industrial_and_Scientific"
TRAIN_FILE=$(ls -f ./data/Amazon/train/${CATEGORY}*11.csv 2>/dev/null || echo "./data/Amazon/train/${CATEGORY}.csv")
EVAL_FILE=$(ls -f ./data/Amazon/valid/${CATEGORY}*11.csv 2>/dev/null || echo "./data/Amazon/valid/${CATEGORY}.csv")
SID_INDEX="./data/Amazon/index/${CATEGORY}.index.json"
ITEM_META="./data/Amazon/index/${CATEGORY}.item.json"

# 检查文件是否存在
echo "检查数据文件..."
echo "训练文件: ${TRAIN_FILE}"
echo "验证文件: ${EVAL_FILE}"
echo "SID 索引: ${SID_INDEX}"
echo "物品元数据: ${ITEM_META}"

# 训练超参数 (针对 4090 24G 优化)
BATCH_SIZE=128         # 总批次大小
MICRO_BATCH_SIZE=16    # 4090 24G 可以支持较大的 batch size
NUM_EPOCHS=3           # 快速验证只训练 3 轮
LEARNING_RATE=5e-4     # 小模型可以用稍大的学习率
CUTOFF_LEN=256         # 序列长度，可根据显存调整到 512
SAMPLE=-1              # -1 使用全部数据，设为 5000-10000 可快速测试

# SwanLab 设置
SWANLAB_PROJECT="MiniOneRec-0.5B"
SWANLAB_RUN_NAME="sft-0.5b-${CATEGORY}-$(date +%m%d_%H%M)"

# 输出目录
OUTPUT_DIR="./outputs/sft_0.5b_${CATEGORY}_$(date +%m%d_%H%M)"

# 训练参数
FREEZE_LLM=False           # 是否冻结 LLM 参数
TRAIN_FROM_SCRATCH=False   # 是否从头训练
SEED=42

echo "========================================="
echo "开始单卡 SFT 训练 (4090 24G + SwanLab)"
echo "基础模型: ${BASE_MODEL}"
echo "数据集: ${CATEGORY}"
echo "批次大小: ${BATCH_SIZE} (micro: ${MICRO_BATCH_SIZE}, 梯度累积: $((BATCH_SIZE/MICRO_BATCH_SIZE)))"
echo "训练轮数: ${NUM_EPOCHS}"
echo "序列长度: ${CUTOFF_LEN}"
echo "输出目录: ${OUTPUT_DIR}"
echo "SwanLab 项目: ${SWANLAB_PROJECT}"
echo "SwanLab 运行名: ${SWANLAB_RUN_NAME}"
echo "========================================="

# 单卡训练 - 不使用 torchrun
python sft.py \
    --base_model ${BASE_MODEL} \
    --batch_size ${BATCH_SIZE} \
    --micro_batch_size ${MICRO_BATCH_SIZE} \
    --num_epochs ${NUM_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --cutoff_len ${CUTOFF_LEN} \
    --train_file ${TRAIN_FILE} \
    --eval_file ${EVAL_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --wandb_project ${SWANLAB_PROJECT} \
    --wandb_run_name ${SWANLAB_RUN_NAME} \
    --use_swanlab True \
    --category ${CATEGORY} \
    --train_from_scratch ${TRAIN_FROM_SCRATCH} \
    --freeze_LLM ${FREEZE_LLM} \
    --seed ${SEED} \
    --sid_index_path ${SID_INDEX} \
    --item_meta_path ${ITEM_META} \
    --sample ${SAMPLE}

echo "========================================="
echo "训练完成！"
echo "模型保存在: ${OUTPUT_DIR}"
echo "查看 SwanLab 结果: https://swanlab.cn/@your-username/${SWANLAB_PROJECT}"
echo "========================================="
