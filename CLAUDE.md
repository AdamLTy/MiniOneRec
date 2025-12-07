# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**MiniOneRec** 是首个完全开源的生成式推荐框架,提供从 **SID(Semantic ID)构建**、**监督微调(SFT)** 到面向推荐的 **强化学习(RL)** 的端到端工作流程。该项目主要针对 Amazon 等电商数据集进行序列推荐任务。

## 环境设置

### 创建环境
```bash
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
```

### 安装依赖
```bash
pip install -r requirements.txt
```

**关键依赖**:
- PyTorch 2.6.0 with CUDA 11.8
- Transformers 4.57.1
- TRL 0.24.0 (用于 GRPO 训练)
- Accelerate 1.10.1 (用于分布式训练)
- DeepSpeed 0.18.0

### 硬件要求
- **推荐配置**: 4-8 × A100/H100 80GB 或同等级 GPU
- **SFT 阶段**: 需要多 GPU (通常使用 8 卡进行 torchrun 训练)
- **RL 阶段**: 使用 Accelerate 进行分布式训练

## 核心工作流程

MiniOneRec 的完整流程分为三个主要阶段:

### 1. 数据准备与预处理

#### 1.1 下载并预处理 Amazon 数据集
```bash
# Amazon 2018 数据集
bash data/amazon18_data_process.sh \
     --dataset Industrial_and_Scientific \
     --user_k 5 \
     --item_k 5 \
     --st_year 1996 \
     --st_month 10 \
     --ed_year 2018 \
     --ed_month 10 \
     --output_path ./data/Amazon18

# Amazon 2023 数据集 (新增)
bash data/amazon23_data_process.sh
```

**参数说明**:
- `--dataset`: 数据集类别 (e.g., Industrial_and_Scientific, Office_Products)
- `--user_k`: 用户最少交互次数
- `--item_k`: 物品最少交互次数
- `--st_year/st_month`: 开始时间
- `--ed_year/ed_month`: 结束时间

#### 1.2 将物品文本转换为嵌入向量
```bash
bash rq/text2emb/amazon_text2emb.sh \
     --dataset Industrial_and_Scientific \
     --root ./data/Amazon18/Industrial_and_Scientific \
     --plm_checkpoint your_emb_model_path
```

**注意**: 新版本使用基于 Accelerate 的多 GPU 并行处理,效率显著提升 (`rq/text2emb/amazon_text2emb.py`)。

### 2. SID 构建 (Semantic ID Construction)

MiniOneRec 提供了四种 SID 构建方法,选择其中一种即可:

#### 方法 1: RQ-VAE (原始方法)
```bash
bash rq/rqvae.sh \
     --data_path ./data/Industrial_and_Scientific/Industrial_and_Scientific.emb-qwen-td.npy \
     --ckpt_dir ./output/Industrial_and_Scientific \
     --lr 1e-3 \
     --epochs 10000 \
     --batch_size 20480

# 生成 SID 索引
python rq/generate_indices.py
```

#### 方法 2: RQ-Kmeans (基于 Faiss)
```bash
conda install faiss-gpu
python rq/rqkmeans_faiss.py --dataset Industrial_and_Scientific
```
**注意**: 基于语义嵌入的 RQ-Kmeans 方法存在较高的碰撞率。

#### 方法 3: Constrained RQ-Kmeans
```bash
pip install k_means_constrained polars
bash rq/rqkmeans_constrained.sh
```
**特点**: 为冲突物品添加额外层进行去重,使用平衡约束确保 SID 均匀分布。

#### 方法 4: RQ-Kmeans+ (推荐,来自 GPR)
```bash
pip install k_means_constrained polars
bash rq/rqkmeans_constrained.sh
bash rq/rqkmeans_plus.sh

# 生成 SID 索引
bash rq/generate_indices_plus.sh
```
**特点**: 首次开源复现 GPR 中提出的方法,性能更优。

#### 2.1 转换数据集格式
```bash
bash convert_dataset.sh
# 或手动运行:
python convert_dataset.py \
     --dataset_name Industrial_and_Scientific \
     --data_dir /path/to/Industrial_and_Scientific \
     --output_dir /path/to/output_dir
```

### 3. 监督微调 (SFT)

```bash
bash sft.sh
```

**关键配置** (编辑 `sft.sh`):
- `--base_model`: 基础 LLM 路径 (e.g., Qwen, Llama)
- `--batch_size`: 总批次大小 (默认 1024)
- `--micro_batch_size`: 每个 GPU 的批次大小 (默认 16)
- `--train_file`, `--eval_file`: 训练/验证数据路径
- `--sid_index_path`: SID 索引文件 (`.index.json`)
- `--item_meta_path`: 物品元数据文件 (`.item.json`)
- `--freeze_LLM`: 是否冻结 LLM 参数,仅训练新增的 SID 词嵌入 (默认 False)
- `--train_from_scratch`: 是否从头开始训练 (默认 False)

**执行方式**: 使用 `torchrun --nproc_per_node 8` 进行多 GPU 训练

**数据集类型** (在 `data.py` 中定义):
- `SidSFTDataset`: 基于 SID 的序列推荐任务
- `SidItemFeatDataset`: SID 与物品特征对齐
- `FusionSeqRecDataset`: 融合多种序列表示
- `PreferenceSFTDataset`: 用户偏好建模
- `UserPreference2sidSFTDataset`: 用户偏好到 SID 的映射
- `TitleHistory2SidSFTDataset`: 标题历史到 SID 的生成

### 4. 强化学习 (RL - GRPO Based)

```bash
bash rl.sh
```

**关键配置** (编辑 `rl.sh`):
- `--model_path`: SFT 阶段训练好的模型路径
- `--train_batch_size`: 训练批次大小 (默认 64)
- `--num_generations`: 每个 prompt 生成的候选数量 (默认 16)
- `--num_train_epochs`: 训练轮数 (默认 2)
- `--reward_type`: 奖励类型 (`rule`, `ranking`)
- `--beam_search`: 是否使用约束束搜索 (默认 True)
- `--beta`: KL 散度惩罚系数 (默认 1e-3)
- `--temperature`: 生成温度 (默认 1.0)
- `--sync_ref_model`: 是否同步参考模型 (默认 True)

**执行方式**: 使用 `accelerate launch --config_file ./config/zero2_opt.yaml --num_processes 8`

**注意**:
- 对于大规模数据集,可仅使用数万样本进行 RL 阶段以降低成本
- `minionerec_trainer.py` 实现了基于 GRPO 的专用训练器
- 支持约束解码 (`LogitProcessor.py`) 以确保生成有效的 SID

### 5. 离线评估

```bash
bash evaluate.sh
```

**评估流程**:
1. **数据分片**: `split.py` 将测试数据分配到多个 GPU
2. **并行推理**: 每个 GPU 独立进行推理,生成推荐结果
3. **结果合并**: `merge.py` 合并所有 GPU 的结果
4. **指标计算**: `calc.py` 计算 HR@K 和 NDCG@K

**关键参数** (编辑 `evaluate.sh`):
- `--base_model`: 待评估的模型路径
- `--batch_size`: 推理批次大小 (默认 8)
- `--num_beams`: 束搜索宽度 (默认 50)
- `--max_new_tokens`: 最大生成 token 数 (默认 256)

## 架构关键点

### 数据管道 (`data.py`)
- **Tokenizer 类**: 封装 tokenizer,处理 BOS/EOS token
- **多种数据集**: 支持不同的训练任务 (SID 生成、标题-SID 对齐、序列推荐等)
- **动态提示**: 使用多样化的指令模板增强泛化能力

### SID 构建 (`rq/` 目录)
- **RQ-VAE** (`rq/models/rqvae.py`): 基于 VAE 的残差量化
  - 多层量化 codebook (默认 3 层,每层 256 个 code)
  - 支持 Sinkhorn 优化和 K-means 初始化
  - 可配置重建损失类型 (MSE, Cosine)

- **RQ-Kmeans 系列**:
  - `rqkmeans_faiss.py`: 基础 Faiss 实现
  - `rqkmeans_constrained.py`: 带平衡约束的版本
  - `rqkmeans_plus.py`: GPR 方法的开源实现

- **索引生成**:
  - `generate_indices.py`: RQ-VAE 索引生成
  - `generate_indices_plus.py`: RQ-Kmeans+ 索引生成

### 训练器 (`minionerec_trainer.py`)
- **ReReTrainer**: 继承自 TRL 的 GRPOTrainer
- **约束解码**: `ConstrainedLogitsProcessor` 确保生成有效 SID
- **奖励函数**: 支持规则奖励和排序奖励
  - `ranking`: 排序感知,对高概率错误物品施加更重惩罚
  - `rule`: 二元正确性奖励
- **自定义采样器**: `RepeatRandomSampler` 支持重复采样策略

### Token 扩展 (`sft.py` 中的 TokenExtender)
- 加载 SID 索引并扩展 tokenizer 词表
- 为新增的 SID token 初始化嵌入
- 支持冻结 LLM 参数仅训练新 token 嵌入

## 常见数据集

**支持的类别** (Amazon 2018):
- `Industrial_and_Scientific`
- `Office_Products`
- `Toys_and_Games`
- `Sports_and_Outdoors`
- `Books`

**数据文件结构**:
```
data/Amazon/
├── train/{category}*.csv         # 训练数据
├── valid/{category}*11.csv       # 验证数据
├── test/{category}*11.csv        # 测试数据
├── info/{category}*.txt          # 物品信息 (semantic_id \t title \t item_id)
└── index/
    ├── {category}.index.json     # SID 索引
    └── {category}.item.json      # 物品元数据
```

## 快速开始 (使用预训练 SID)

如果想跳过 SID 构建阶段,可以直接下载预构建的 Industrial/Office SID:

```bash
# 1. 环境设置
conda create -n MiniOneRec python=3.11 -y
conda activate MiniOneRec
pip install -r requirements.txt

# 2. 下载预训练 SID (从 Huggingface/Modelscope)

# 3. SFT
bash sft.sh

# 4. RL
bash rl.sh

# 5. 评估
bash evaluate.sh
```

## 配置文件

- **DeepSpeed 配置**: `config/zero2_opt.yaml` - ZeRO Stage 2 优化配置,用于 RL 阶段
- **Gin 配置**: 项目主要使用 shell 脚本传参,未使用 Gin 配置系统

## 重要提示

1. **数据处理 Bug 修复** (2025-12-01): `data.py` 修复了 SID-item 对齐任务提前泄露答案的 bug

2. **NCCL 设置**: shell 脚本中使用 `export NCCL_IB_DISABLE=1` 禁用 InfiniBand/RoCE,适配不支持的环境

3. **随机种子**: 所有脚本默认使用 `seed=42` 保证可复现性

4. **分布式训练**:
   - SFT: 使用 `torchrun`
   - RL: 使用 `accelerate launch`
   - 确保在多 GPU 环境下正确配置进程数和端口

5. **约束束搜索**: RL 阶段使用约束束搜索保证生成的 beam 唯一且有效,显著提升采样效率和多样性

6. **评估并行化**: 使用多 GPU 并行推理 + 结果合并的方式加速大规模评估

## 引用和致谢

本项目复用或改编了以下开源项目的代码:
- [ReRe](https://github.com/sober-clever/ReRe)
- [LC-Rec](https://github.com/zhengbw0324/LC-Rec)
