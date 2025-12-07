SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
accelerate launch --num_processes 8 "${SCRIPT_DIR}/amazon_text2emb.py" \
    --dataset Industrial_and_Scientific \
    --root ../../data/Amazon18/Industrial_and_Scientific \
    --plm_checkpoint your_emb_model_path
