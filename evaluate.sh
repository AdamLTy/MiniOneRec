#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# 支持通过参数传递配置
BASE_MODEL="${BASE_MODEL:-${1:-xxx}}"
CATEGORY="${CATEGORY:-${2:-Industrial_and_Scientific}}"
BATCH_SIZE="${BATCH_SIZE:-${3:-8}}"
NUM_BEAMS="${NUM_BEAMS:-${4:-50}}"

# Industrial_and_Scientific
# Office_Products
for category in "$CATEGORY"
do
    # your model path
    exp_name="$BASE_MODEL"

    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"

    train_file=$(ls "${PROJECT_ROOT}/data/Amazon/train/${category}"*.csv 2>/dev/null | head -1)
    test_file=$(ls "${PROJECT_ROOT}/data/Amazon/test/${category}"*11.csv 2>/dev/null | head -1)
    info_file=$(ls "${PROJECT_ROOT}/data/Amazon/info/${category}"*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    temp_dir="${PROJECT_ROOT}/temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"

    echo "Splitting test data..."
    python "${PROJECT_ROOT}/split.py" --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3,4,5,6,7"
    
    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi
    
    cudalist="0 1 2 3 4 5 6 7"  
    echo "Starting parallel evaluation (STANDARD MODE)..."
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            CUDA_VISIBLE_DEVICES=$i python -u "${PROJECT_ROOT}/evaluate.py" \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size "$BATCH_SIZE" \
                --num_beams "$NUM_BEAMS" \
                --max_new_tokens 256 \
                --temperature 1.0 \
                --guidance_scale 1.0 \
                --length_penalty 0.0 &
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    wait
    
    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi
    
    output_dir="${PROJECT_ROOT}/results/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"

    python "${PROJECT_ROOT}/merge.py" \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"

    if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi

    echo "Calculating metrics..."
    python "${PROJECT_ROOT}/calc.py" \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------" 
done

echo "All categories processed!"
