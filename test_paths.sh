#!/bin/bash
# 路径修复验证脚本

echo "=========================================="
echo "MiniOneRec 路径修复验证测试"
echo "=========================================="
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PASS_COUNT=0
FAIL_COUNT=0

# 测试函数
test_script() {
    local script_name=$1
    local test_desc=$2

    echo "测试: $test_desc"
    echo "脚本: $script_name"

    if [ -f "$script_name" ]; then
        # 检查脚本是否有 SCRIPT_DIR 变量定义
        if grep -q "SCRIPT_DIR=" "$script_name"; then
            echo "✓ 脚本包含 SCRIPT_DIR 定义"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "✗ 脚本缺少 SCRIPT_DIR 定义"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "✗ 脚本文件不存在"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

# 测试 Python 文件是否有路径处理
test_python() {
    local py_file=$1
    local test_desc=$2
    local search_pattern=$3

    echo "测试: $test_desc"
    echo "文件: $py_file"

    if [ -f "$py_file" ]; then
        if grep -q "$search_pattern" "$py_file"; then
            echo "✓ Python 文件包含 $search_pattern"
            PASS_COUNT=$((PASS_COUNT + 1))
        else
            echo "✗ Python 文件缺少 $search_pattern"
            FAIL_COUNT=$((FAIL_COUNT + 1))
        fi
    else
        echo "✗ Python 文件不存在"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
    echo ""
}

echo "==================== Shell 脚本测试 ===================="
test_script "data/amazon18_data_process.sh" "数据处理脚本路径修复"
test_script "rq/text2emb/amazon_text2emb.sh" "文本转嵌入脚本路径修复"
test_script "convert_dataset.sh" "数据集转换脚本路径修复"
test_script "evaluate.sh" "评估脚本路径修复"

echo "==================== Python 脚本测试 ===================="
test_python "rq/rqvae.py" "RQ-VAE 导入路径修复" "sys.path.insert"
test_python "rq/generate_indices.py" "索引生成参数化" "argparse"
test_python "sft.py" "SFT 路径验证" "FileNotFoundError"
test_python "rl.py" "RL 路径验证" "FileNotFoundError"

echo "==================== 测试总结 ===================="
echo "通过: $PASS_COUNT"
echo "失败: $FAIL_COUNT"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo "✓✓✓ 所有测试通过! ✓✓✓"
    exit 0
else
    echo "✗✗✗ 有 $FAIL_COUNT 个测试失败 ✗✗✗"
    exit 1
fi
