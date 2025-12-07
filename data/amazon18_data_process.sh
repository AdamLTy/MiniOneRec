SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

python "${SCRIPT_DIR}/amazon18_data_process.py" \
    --dataset "${DATASET:-Industrial_and_Scientific}" \
    --user_k "${USER_K:-5}" \
    --item_k "${ITEM_K:-5}" \
    --st_year "${ST_YEAR:-1996}" \
    --st_month "${ST_MONTH:-10}" \
    --ed_year "${ED_YEAR:-2018}" \
    --ed_month "${ED_MONTH:-10}" \
    --output_path "${OUTPUT_PATH:-${PROJECT_ROOT}/data/Amazon18}"
