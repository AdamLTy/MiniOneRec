#!/bin/bash
# MiniOneRec 0.5B 完整训练流程
# 从 SID 构建到强化学习的端到端执行脚本

set -e  # 遇到错误立即退出

echo "=================================================="
echo "  MiniOneRec 0.5B Baseline 完整训练流程"
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
echo ""

# ==================== 阶段 1: 数据下载与预处理 ====================
echo "【阶段 1/7】 数据下载与预处理"
echo "--------------------------------------------------"

if [ ! -d "${DATA_ROOT}" ]; then
    echo "开始下载和预处理 Amazon 数据集..."
    bash data/amazon18_data_process.sh \
        --dataset ${CATEGORY} \
        --user_k 5 \
        --item_k 5 \
        --st_year 1996 \
        --st_month 10 \
        --ed_year 2018 \
        --ed_month 10 \
        --output_path ./data/Amazon18
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
    bash rq/text2emb/amazon_text2emb.sh \
        --dataset ${CATEGORY} \
        --root ${DATA_ROOT} \
        --plm_checkpoint ${EMB_MODEL}
else
    echo "嵌入文件已存在,跳过生成"
fi

# ==================== 阶段 3: SID 构建 (RQ-Kmeans+) ====================
echo ""
echo "【阶段 3/7】 SID 构建 (推荐使用 RQ-Kmeans+)"
echo "--------------------------------------------------"

SID_OUTPUT_DIR="${OUTPUT_ROOT}/sid"
mkdir -p ${SID_OUTPUT_DIR}

echo "选择 SID 构建方法:"
echo "1) RQ-Kmeans+ (推荐,来自 GPR)"
echo "2) Constrained RQ-Kmeans"
echo "3) RQ-VAE (原始方法)"
read -p "请选择 [1-3, 默认 1]: " sid_method
sid_method=${sid_method:-1}

case $sid_method in
    1)
        echo "使用 RQ-Kmeans+ 方法..."
        # 先运行 Constrained RQ-Kmeans
        bash rq/rqkmeans_constrained.sh \
            --dataset ${CATEGORY} \
            --data_path ${EMB_FILE} \
            --output_dir ${SID_OUTPUT_DIR}/constrained

        # 再运行 Plus 优化
        bash rq/rqkmeans_plus.sh \
            --dataset ${CATEGORY} \
            --constrained_output ${SID_OUTPUT_DIR}/constrained \
            --output_dir ${SID_OUTPUT_DIR}/plus

        # 生成索引
        bash rq/generate_indices_plus.sh \
            --dataset ${CATEGORY} \
            --sid_dir ${SID_OUTPUT_DIR}/plus \
            --data_root ${DATA_ROOT}
        ;;
    2)
        echo "使用 Constrained RQ-Kmeans 方法..."
        bash rq/rqkmeans_constrained.sh \
            --dataset ${CATEGORY} \
            --data_path ${EMB_FILE} \
            --output_dir ${SID_OUTPUT_DIR}/constrained
        ;;
    3)
        echo "使用 RQ-VAE 方法 (带 SwanLab 监控)..."
        cd rq
        python rqvae.py \
            --data_path ../${EMB_FILE} \
            --ckpt_dir ../${SID_OUTPUT_DIR}/rqvae \
            --lr 1e-3 \
            --epochs 10000 \
            --batch_size 20480 \
            --eval_step 50 \
            --num_emb_list 256 256 256 \
            --e_dim 32 \
            --use_swanlab True \
            --swanlab_project "MiniOneRec-Pipeline-RQVAE" \
            --swanlab_run_name "rqvae-${CATEGORY}-$(date +%m%d_%H%M)"
        cd ..

        python rq/generate_indices.py \
            --dataset ${CATEGORY} \
            --rqvae_dir ${SID_OUTPUT_DIR}/rqvae \
            --data_root ${DATA_ROOT}
        ;;
esac

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
echo "【阶段 5/7】 监督微调 (SFT)"
echo "--------------------------------------------------"

SFT_OUTPUT="${OUTPUT_ROOT}/sft_model"

python sft.py \
    --base_model ${BASE_MODEL} \
    --batch_size 128 \
    --micro_batch_size 16 \
    --num_epochs 3 \
    --learning_rate 5e-4 \
    --cutoff_len 256 \
    --train_file ./data/Amazon/train/${CATEGORY}*11.csv \
    --eval_file ./data/Amazon/valid/${CATEGORY}*11.csv \
    --output_dir ${SFT_OUTPUT} \
    --wandb_project "MiniOneRec-0.5B-Baseline" \
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
echo "【阶段 6/7】 强化学习 (GRPO)"
echo "--------------------------------------------------"

RL_OUTPUT="${OUTPUT_ROOT}/rl_model"

# 询问是否跳过 RL 阶段
read -p "是否跳过 RL 阶段? [y/N]: " skip_rl
skip_rl=${skip_rl:-N}

if [[ "$skip_rl" =~ ^[Yy]$ ]]; then
    echo "跳过 RL 阶段"
    FINAL_MODEL=${SFT_OUTPUT}
else
    echo "开始 RL 训练..."

    # 对于快速验证,可以只使用部分数据
    read -p "仅使用 10k 样本进行快速 RL? [Y/n]: " use_sample
    use_sample=${use_sample:-Y}

    if [[ "$use_sample" =~ ^[Yy]$ ]]; then
        SAMPLE_FLAG="--sample_train True --sample_size 10000"
    else
        SAMPLE_FLAG="--sample_train False"
    fi

    python rl.py \
        --model_path ${SFT_OUTPUT} \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --num_train_epochs 1 \
        --gradient_accumulation_steps 2 \
        --train_file ./data/Amazon/train/${CATEGORY}*.csv \
        --eval_file ./data/Amazon/valid/${CATEGORY}*11.csv \
        --info_file ./data/Amazon/info/${CATEGORY}*.txt \
        --category ${CATEGORY} \
        ${SAMPLE_FLAG} \
        --eval_step 0.1 \
        --reward_type ranking \
        --num_generations 16 \
        --beam_search True \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --beta 1e-3 \
        --output_dir ${RL_OUTPUT} \
        --wandb_run_name "rl-${CATEGORY}-$(date +%m%d_%H%M)" \
        --sid_index_path ./data/Amazon/index/${CATEGORY}.index.json \
        --item_meta_path ./data/Amazon/index/${CATEGORY}.item.json

    FINAL_MODEL=${RL_OUTPUT}
fi

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
echo "SFT 模型: ${SFT_OUTPUT}"
echo "RL 模型: ${RL_OUTPUT}"
echo "评估结果: ${EVAL_OUTPUT}"
echo ""
echo "下一步: 运行 Bad Case 分析"
echo "  bash analyze_badcase.sh --model ${FINAL_MODEL} --category ${CATEGORY}"
echo "=================================================="
