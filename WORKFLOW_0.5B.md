# MiniOneRec 0.5B Baseline 完整工作流程

## 目录
1. [环境准备](#环境准备)
2. [完整训练流程](#完整训练流程)
3. [Bad Case 分析](#bad-case-分析)
4. [优化策略](#优化策略)
5. [常见问题](#常见问题)

---

## 环境准备

### 1. 创建虚拟环境
```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 准备预训练模型
下载以下模型到本地:
- **基础 LLM**: `Qwen/Qwen2.5-0.5B-Instruct` (推荐) 或 `meta-llama/Llama-3.2-1B-Instruct`
- **文本嵌入模型**: `BAAI/bge-small-en-v1.5`

---

## 完整训练流程

### 方式 1: 一键执行 (推荐用于首次运行)

```bash
bash pipeline_0.5b.sh
```

该脚本会自动执行以下所有步骤,并在每个阶段提供交互式选项。

### 方式 2: 分步执行 (推荐用于调试和定制)

#### 步骤 1: 数据下载与预处理
```bash
cd data
bash amazon18_data_process.sh \
    --dataset Industrial_and_Scientific \
    --user_k 5 \
    --item_k 5 \
    --st_year 1996 \
    --st_month 10 \
    --ed_year 2018 \
    --ed_month 10 \
    --output_path ./Amazon18
```

**输出**: `./data/Amazon18/Industrial_and_Scientific/`

#### 步骤 2: 文本转嵌入
```bash
bash rq/text2emb/amazon_text2emb.sh \
    --dataset Industrial_and_Scientific \
    --root ./data/Amazon18/Industrial_and_Scientific \
    --plm_checkpoint /path/to/bge-small-en-v1.5
```

**输出**: `Industrial_and_Scientific.emb.npy`

#### 步骤 3: SID 构建

**推荐方法: RQ-Kmeans+ (最优性能)**
```bash
# 先安装依赖
pip install k_means_constrained polars

# 运行 Constrained RQ-Kmeans
bash rq/rqkmeans_constrained.sh

# 运行 Plus 优化
bash rq/rqkmeans_plus.sh

# 生成索引
bash rq/generate_indices_plus.sh
```

**其他方法**:
- **RQ-VAE**: `bash rq/rqvae.sh`
- **RQ-Kmeans**: `python rq/rqkmeans_faiss.py`

**输出**:
- `./data/Amazon/index/Industrial_and_Scientific.index.json`
- `./data/Amazon/index/Industrial_and_Scientific.item.json`

#### 步骤 4: 数据集转换
```bash
bash convert_dataset.sh
```

**输出**: `./data/Amazon/train/`, `./data/Amazon/valid/`, `./data/Amazon/test/`

#### 步骤 5: SFT 训练 (单卡)

**使用提供的单卡脚本**:
```bash
bash sft_single_gpu_swanlab.sh
```

**关键参数**:
- `BASE_MODEL`: 基础模型路径
- `BATCH_SIZE`: 128 (总批次大小)
- `MICRO_BATCH_SIZE`: 16 (单卡批次)
- `NUM_EPOCHS`: 3
- `LEARNING_RATE`: 5e-4
- `CUTOFF_LEN`: 256

**训练时间估计** (单卡 A100 80G):
- Industrial_and_Scientific: ~2-3 小时/epoch
- 总计: ~6-9 小时

**输出**: `./outputs/sft_0.5b_{category}_{timestamp}/`

#### 步骤 6: RL 训练 (可选)

**快速验证 (仅 10k 样本)**:
```bash
export CUDA_VISIBLE_DEVICES=0

python rl.py \
    --model_path ./outputs/sft_0.5b_Industrial_and_Scientific_xxx \
    --train_batch_size 32 \
    --num_train_epochs 1 \
    --sample_train True \
    --sample_size 10000 \
    --reward_type ranking \
    --num_generations 16 \
    ... (其他参数见 rl.sh)
```

**完整 RL 训练**:
```bash
bash rl.sh
```

**训练时间估计**:
- 10k 样本: ~1-2 小时
- 完整数据集: ~10-15 小时

**输出**: `./outputs/rl_model/`

#### 步骤 7: 模型评估
```bash
bash evaluate.sh \
    --base_model ./outputs/sft_0.5b_Industrial_and_Scientific_xxx \
    --category Industrial_and_Scientific
```

**评估指标**:
- HR@5, HR@10, HR@20
- NDCG@5, NDCG@10, NDCG@20

**输出**: `./evaluation/results.json`

---

## Bad Case 分析

### 1. 运行分析脚本
```bash
bash analyze_badcase.sh \
    --model ./outputs/sft_0.5b_Industrial_and_Scientific_xxx \
    --category Industrial_and_Scientific \
    --top_k 10
```

### 2. 分析流程
该脚本会自动完成:
1. **运行评估**: 在测试集上生成预测结果
2. **识别 Bad Cases**: 找出 Top-K 推荐失败的样本
3. **模式分析**: 分析失败原因 (冷启动、长尾、短历史等)
4. **SID 质量分析**: 检测 SID 碰撞、分布不均等问题
5. **生成可视化**: 失败模式分布图、排名分布图等

### 3. 输出文件
```
./analysis_output/{category}_{timestamp}/
├── eval_results.jsonl          # 评估结果 (预测 + 真实标签)
├── badcase_report.json         # 分析报告
├── visualizations/
│   ├── failure_patterns.png    # 失败模式分布
│   ├── failure_rate.png        # 失败率饼图
│   ├── sid_quality.png         # SID 质量分析
│   ├── rank_distribution.png   # 真实物品排名分布
│   └── summary_report.txt      # 文本摘要报告
```

### 4. 查看分析结果
```bash
# 查看文本摘要
cat ./analysis_output/{category}_{timestamp}/visualizations/summary_report.txt

# 查看详细报告
cat ./analysis_output/{category}_{timestamp}/badcase_report.json | jq
```

---

## 优化策略

### 1. 自动推荐优化策略
```bash
python optimization_strategies.py \
    --badcase_report ./analysis_output/{category}_{timestamp}/badcase_report.json \
    --output_config ./optimized_config.json \
    --auto_recommend
```

### 2. 手动选择策略
```bash
python optimization_strategies.py \
    --badcase_report ./analysis_output/{category}_{timestamp}/badcase_report.json \
    --output_config ./optimized_config.json
```

### 3. 可用的优化策略

| 策略名称 | 适用场景 | 主要改进 |
|---------|---------|---------|
| `long_tail` | 长尾物品推荐失败率高 | 频率平衡采样、多样性奖励 |
| `cold_start` | 冷启动物品效果差 | 物品特征对齐、标题嵌入 |
| `short_history` | 短历史用户准确率低 | 用户画像、协同过滤增强 |
| `sid_quality` | SID 碰撞或质量问题 | 增加量化层、碰撞检测 |
| `recency_bias` | 忽略最近偏好 | 位置编码、时间衰减 |
| `hyperparameter` | 通用优化 | 调整学习率、正则化 |

### 4. 应用优化策略重新训练

**查看生成的配置**:
```bash
cat optimized_config.json
```

**根据优化建议修改训练脚本,然后重新训练**:
```bash
# 修改 sft_single_gpu_swanlab.sh 中的参数
# 例如: 增加训练轮数、调整学习率、启用新特征等

bash sft_single_gpu_swanlab.sh
```

### 5. 对比优化前后效果
```bash
# 评估优化前的模型
bash evaluate.sh --base_model ./outputs/baseline_model --category Industrial_and_Scientific

# 评估优化后的模型
bash evaluate.sh --base_model ./outputs/optimized_model --category Industrial_and_Scientific

# 对比指标
python compare_results.py \
    --baseline ./evaluation/baseline_results.json \
    --optimized ./evaluation/optimized_results.json
```

---

## 完整工作流程图

```
┌─────────────────────────────────────────────────────────────┐
│                     【阶段 1: 数据准备】                      │
├─────────────────────────────────────────────────────────────┤
│  1. 下载 Amazon 数据集                                       │
│  2. 预处理 (过滤低频用户/物品)                               │
│  3. 物品文本 → 嵌入向量                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     【阶段 2: SID 构建】                      │
├─────────────────────────────────────────────────────────────┤
│  选择方法:                                                   │
│  ├─ RQ-Kmeans+ (推荐)                                       │
│  ├─ Constrained RQ-Kmeans                                   │
│  └─ RQ-VAE                                                  │
│                                                             │
│  输出: SID 索引 (.index.json, .item.json)                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    【阶段 3: SFT 训练】                       │
├─────────────────────────────────────────────────────────────┤
│  - 基础模型: Qwen 0.5B                                       │
│  - 训练轮数: 3 epochs                                        │
│  - 任务: 序列推荐 (历史 → SID)                               │
│  - 监控: SwanLab 实时跟踪                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    【阶段 4: RL 优化】                        │
├─────────────────────────────────────────────────────────────┤
│  - 算法: GRPO                                               │
│  - 奖励: Ranking-aware                                      │
│  - 约束: 束搜索保证有效性                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    【阶段 5: 评估】                           │
├─────────────────────────────────────────────────────────────┤
│  - 指标: HR@K, NDCG@K                                        │
│  - 保存: 预测结果 + 真实标签                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                【阶段 6: Bad Case 分析】                      │
├─────────────────────────────────────────────────────────────┤
│  1. 识别失败样本 (GT 不在 Top-K)                             │
│  2. 分析失败模式:                                            │
│     - 冷启动、长尾、短历史                                   │
│     - 类别不匹配、忽略最近偏好                               │
│  3. SID 质量检查:                                            │
│     - 碰撞检测、分布均衡性                                   │
│  4. 生成可视化报告                                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                【阶段 7: 优化策略制定】                       │
├─────────────────────────────────────────────────────────────┤
│  1. 自动推荐优化策略                                         │
│  2. 生成优化配置文件                                         │
│  3. 应用优化策略:                                            │
│     - 调整数据采样                                           │
│     - 增强特征工程                                           │
│     - 改进奖励函数                                           │
│     - 优化超参数                                             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                【阶段 8: 迭代优化】                           │
├─────────────────────────────────────────────────────────────┤
│  1. 应用优化策略重新训练                                     │
│  2. 评估新模型                                               │
│  3. 对比优化前后效果                                         │
│  4. 继续分析 Bad Cases (回到阶段 6)                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 常见问题

### Q1: 训练时显存不足怎么办?
**A**: 调整以下参数:
- 减小 `MICRO_BATCH_SIZE` (如 16 → 8)
- 减小 `CUTOFF_LEN` (如 256 → 128)
- 启用梯度检查点: `--gradient_checkpointing True`
- 使用 DeepSpeed ZeRO-2: 修改 `config/zero2_opt.yaml`

### Q2: SID 碰撞率高怎么办?
**A**:
1. 使用 RQ-Kmeans+ 方法替代其他方法
2. 增加量化层数: `--num_quantization_layers 4`
3. 增加 codebook 大小: `--codebook_size 512`
4. 使用 Constrained RQ-Kmeans 添加额外去重层

### Q3: 如何加速 RL 训练?
**A**:
1. **仅使用部分数据**: `--sample_train True --sample_size 10000`
2. **减少候选生成数**: `--num_generations 8` (默认 16)
3. **降低束宽**: 生成时使用较小的 beam size
4. **跳过频繁评估**: `--eval_step 0.2` (仅在 20% 和结束时评估)

### Q4: 如何选择 SID 构建方法?
**A**:
- **首次运行**: RQ-Kmeans+ (最优性能,已验证)
- **快速验证**: RQ-Kmeans (最快,但碰撞率高)
- **研究对比**: RQ-VAE (原始方法,训练时间长)

### Q5: RL 阶段是否必须?
**A**:
- **仅 SFT** 即可获得 baseline 结果
- **SFT + RL** 可进一步提升 2-5% 指标
- 如果时间/资源有限,可跳过 RL,专注于优化 SFT 阶段

### Q6: 如何解读 Bad Case 分析结果?
**A**: 重点关注:
1. **失败率**: 如果 >30%,说明模型需要大幅改进
2. **主要失败模式**: 占比 >20% 的模式需要针对性优化
3. **真实物品排名**: 如果大部分在 20-50 名,说明模型方向正确但需微调
4. **SID 碰撞数**: 如果 >0,优先解决 SID 质量问题

---

## 预期结果

### Industrial_and_Scientific 数据集

| 模型 | HR@5 | HR@10 | NDCG@10 |
|------|------|-------|---------|
| SASRec (Baseline) | ~8% | ~12% | ~6% |
| MiniOneRec SFT (0.5B) | ~15-18% | ~22-25% | ~12-15% |
| MiniOneRec SFT+RL (0.5B) | ~18-22% | ~25-30% | ~15-18% |
| MiniOneRec Optimized | ~20-24% | ~28-32% | ~17-20% |

*注: 具体结果取决于数据预处理、SID 质量和训练超参数*

---

## 下一步建议

完成 0.5B baseline 后,可以尝试:

1. **扩展到更大模型**:
   - Qwen 1.5B / 3B
   - Llama 3B / 7B

2. **多数据集验证**:
   - Office_Products
   - Toys_and_Games
   - Sports_and_Outdoors

3. **高级优化**:
   - 多任务学习 (序列推荐 + 物品理解)
   - 元学习快速适应新用户
   - 持续学习处理数据漂移

4. **生产部署**:
   - 模型量化 (INT8/INT4)
   - 推理加速 (TensorRT/ONNX)
   - 在线 A/B 测试

---

**祝训练顺利! 🚀**
