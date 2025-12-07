#!/bin/bash
# MiniOneRec 0.5B 完整训练流程 (4090 24G 单卡优化版)
# 从 SID 构建到强化学习的端到端执行脚本

set -e  # 遇到错误立即退出

echo "=================================================="
echo "  MiniOneRec 0.5B Baseline - 4090 24G 优化版"
echo "=================================================="

# ==================== 配置参数 ====================
CATEGORY="Industrial_and_Scientific"  # 可选: Office_Products, Toys_and_Games
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"  # 修改为你的本地模型路径
EMB_MODEL="BAAI/bge-small-en-v1.5"  # 文本嵌入模型

# 数据路径
DATA_ROOT="./data/Amazon18/${CATEGORY}"
OUTPUT_ROOT="./outputs/baseline_0.5b_$(date +%Y%m%d)"

# GPU 配置
export CUDA_VISIBLE_DEVICES=0
export NCCL_IB_DISABLE=1

echo "数据集: ${CATEGORY}"
echo "基础模型: ${BASE_MODEL}"
echo "输出目录: ${OUTPUT_ROOT}"
echo "硬件配置: 4090 24G 单卡"
echo ""

# ==================== 阶段 1: 数据下载与预处理 ====================
echo "【阶段 1/7】 数据下载与预处理"
echo "--------------------------------------------------"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "开始下载和预处理 Amazon 数据集..."
    DATASET=${CATEGORY} \
    USER_K=5 \
    ITEM_K=5 \
    ST_YEAR=1996 \
    ST_MONTH=10 \
    ED_YEAR=2018 \
    ED_MONTH=11 \
    OUTPUT_PATH=./data/Amazon18 \
    bash data/amazon18_data_process.sh
else
    echo "数据集已存在,跳过下载"
fi

# ==================== 阶段 2: 文本转嵌入 ====================
echo ""
echo "【阶段 2/7】 物品文本转嵌入向量"
echo "--------------------------------------------------"

EMB_FILE="${DATA_ROOT}/${CATEGORY}.emb.npy"
if [ ! -f "${EMB_FILE}" ]; then
    echo "开始生成物品嵌入..."
    DATASET=${CATEGORY} \
    ROOT=${DATA_ROOT} \
    PLM_CHECKPOINT=${EMB_MODEL} \
    NUM_PROCESSES=8 \
    bash rq/text2emb/amazon_text2emb.sh
else
    echo "嵌入文件已存在,跳过生成"
fi

# ==================== 阶段 3: SID 构建 (RQ-VAE) ====================
echo ""
echo "【阶段 3/7】 SID 构建 (使用 RQ-VAE 方法)"
echo "--------------------------------------------------"

SID_OUTPUT_DIR="${OUTPUT_ROOT}/sid"
mkdir -p ${SID_OUTPUT_DIR}

echo "使用 RQ-VAE 方法 (带 SwanLab 监控)..."
python rq/rqvae.py \
    --data_path ${EMB_FILE} \
    --ckpt_dir ${SID_OUTPUT_DIR}/rqvae \
    --lr 1e-3 \
    --epochs 10000 \
    --batch_size 20480 \
    --eval_step 50 \
    --num_emb_list 256 256 256 \
    --e_dim 32 \
    --use_swanlab True \
    --swanlab_project "MiniOneRec-Pipeline-RQVAE" \
    --swanlab_run_name "rqvae-${CATEGORY}-$(date +%m%d_%H%M)"

python rq/generate_indices.py \
    --dataset ${CATEGORY} \
    --rqvae_dir ${SID_OUTPUT_DIR}/rqvae \
    --data_root ${DATA_ROOT}

# ==================== 阶段 4: 数据集转换 ====================
echo ""
echo "【阶段 4/7】 转换数据集格式"
echo "--------------------------------------------------"

bash convert_dataset.sh \
    --dataset_name ${CATEGORY} \
    --data_dir ${DATA_ROOT} \
    --output_dir ./data/Amazon

# ==================== 阶段 5: SFT 训练 ====================
echo ""
echo "【阶段 5/7】 监督微调 (SFT) - 4090 24G 优化配置"
echo "--------------------------------------------------"

SFT_OUTPUT="${OUTPUT_ROOT}/sft_model"

# 4090 24G 优化配置:
# - batch_size 512 (有效批次大小)
# - micro_batch_size 8 (单卡批次,降低以适配显存)
# - gradient_accumulation_steps 64 (8 * 64 = 512)
# - num_epochs 5 (增加训练轮数)
# - bf16=True (代码中已启用,降低显存占用)

python sft.py \
    --base_model ${BASE_MODEL} \
    --batch_size 512 \
    --micro_batch_size 8 \
    --num_epochs 5 \
    --learning_rate 5e-4 \
    --cutoff_len 256 \
    --train_file ./data/Amazon/train/${CATEGORY}*11.csv \
    --eval_file ./data/Amazon/valid/${CATEGORY}*11.csv \
    --output_dir ${SFT_OUTPUT} \
    --wandb_project "MiniOneRec-0.5B-Baseline-4090" \
    --wandb_run_name "sft-${CATEGORY}-$(date +%m%d_%H%M)" \
    --use_swanlab True \
    --category ${CATEGORY} \
    --train_from_scratch False \
    --freeze_LLM False \
    --seed 42 \
    --sid_index_path ./data/Amazon/index/${CATEGORY}.index.json \
    --item_meta_path ./data/Amazon/index/${CATEGORY}.item.json \
    --sample -1

echo "SFT 训练完成,模型保存在: ${SFT_OUTPUT}"

# ==================== 阶段 6: 强化学习 (RL) ====================
echo ""
echo "【阶段 6/7】 强化学习 (GRPO) - 4090 24G 优化配置"
echo "--------------------------------------------------"

RL_OUTPUT="${OUTPUT_ROOT}/rl_model"

# 4090 24G 优化配置:
# - sample_size 50000 (从10k增加到50k,保证训练充分)
# - train_batch_size 16 (降低批次以适配单卡显存)
# - gradient_accumulation_steps 4 (有效批次 16*4=64)
# - num_train_epochs 2 (增加训练轮数)
# - num_generations 16 (保持多样性)
# - reward_type ranking (排序感知奖励)

echo "开始 RL 训练 (使用 50k 样本,2 epochs)..."

python rl.py \
    --model_path ${SFT_OUTPUT} \
    --train_batch_size 16 \
    --eval_batch_size 32 \
    --num_train_epochs 2 \
    --gradient_accumulation_steps 4 \
    --train_file ./data/Amazon/train/${CATEGORY}*.csv \
    --eval_file ./data/Amazon/valid/${CATEGORY}*11.csv \
    --info_file ./data/Amazon/info/${CATEGORY}*.txt \
    --category ${CATEGORY} \
    --sample_train True \
    --sample_size 50000 \
    --eval_step 0.1 \
    --reward_type ranking \
    --num_generations 16 \
    --beam_search True \
    --temperature 1.0 \
    --learning_rate 1e-5 \
    --beta 1e-3 \
    --sync_ref_model True \
    --output_dir ${RL_OUTPUT} \
    --wandb_run_name "rl-${CATEGORY}-$(date +%m%d_%H%M)" \
    --sid_index_path ./data/Amazon/index/${CATEGORY}.index.json \
    --item_meta_path ./data/Amazon/index/${CATEGORY}.item.json

FINAL_MODEL=${RL_OUTPUT}

# ==================== 阶段 7: 模型评估 ====================
echo ""
echo "【阶段 7/7】 模型评估"
echo "--------------------------------------------------"

EVAL_OUTPUT="${OUTPUT_ROOT}/evaluation"
mkdir -p ${EVAL_OUTPUT}

echo "开始在测试集上评估模型..."
bash evaluate.sh \
    --base_model ${FINAL_MODEL} \
    --test_file ./data/Amazon/test/${CATEGORY}*11.csv \
    --batch_size 8 \
    --num_beams 50 \
    --output_dir ${EVAL_OUTPUT} \
    --category ${CATEGORY}

echo ""
echo "=================================================="
echo "  训练流程全部完成!"
echo "=================================================="
echo "硬件配置: 4090 24G 单卡"
echo "SFT 模型: ${SFT_OUTPUT}"
echo "  - 有效批次大小: 512 (8 micro × 64 accumulation)"
echo "  - 训练轮数: 5 epochs"
echo "RL 模型: ${RL_OUTPUT}"
echo "  - 训练样本: 50,000"
echo "  - 有效批次大小: 64 (16 batch × 4 accumulation)"
echo "  - 训练轮数: 2 epochs"
echo "评估结果: ${EVAL_OUTPUT}"
echo ""
echo "下一步: 运行 Bad Case 分析"
echo "  bash analyze_badcase.sh --model ${FINAL_MODEL} --category ${CATEGORY}"
echo "=================================================="
