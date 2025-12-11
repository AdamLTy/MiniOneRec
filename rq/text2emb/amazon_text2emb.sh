accelerate launch --num_processes 1 amazon_text2emb.py \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon18/Industrial_and_Scientific \
    --plm_checkpoint Qwen/Qwen2.5-0.5B-Instruct
