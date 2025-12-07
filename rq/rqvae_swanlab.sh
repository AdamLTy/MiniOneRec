#!/bin/bash
# RQ-VAE 训练脚本 (带 SwanLab 监控)

CATEGORY="Industrial_and_Scientific"
DATA_PATH="../data/Amazon/index/${CATEGORY}.emb.npy"
CKPT_DIR="./output/${CATEGORY}_swanlab"

python rqvae.py \
    --data_path ${DATA_PATH} \
    --ckpt_dir ${CKPT_DIR} \
    --lr 1e-3 \
    --epochs 10000 \
    --batch_size 20480 \
    --eval_step 50 \
    --num_emb_list 256 256 256 \
    --e_dim 32 \
    --layers 2048 1024 512 256 128 64 \
    --use_swanlab True \
    --swanlab_project "MiniOneRec-RQVAE" \
    --swanlab_run_name "rqvae-${CATEGORY}-$(date +%m%d_%H%M)"

echo "训练完成! 查看 SwanLab: https://swanlab.cn"
