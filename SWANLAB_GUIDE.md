# SwanLab 集成指南 - MiniOneRec

## 什么是 SwanLab？

SwanLab 是一款国产开源的实验跟踪与可视化工具，类似于 Weights & Biases (wandb)，但更适合国内用户：
- 🚀 **国内访问快速稳定**，无需科学上网
- 💰 **完全免费**，无使用限制
- 🔒 **支持私有部署**，数据安全可控
- 🎯 **中文界面友好**，易于上手

官网：https://swanlab.cn

## 快速开始

### 步骤 1: 安装 SwanLab

```bash
conda activate MiniOneRec
pip install swanlab
swanlab --version
```

### 步骤 2: 登录

```bash
# 方法 1: 交互式登录（推荐）
swanlab login

# 方法 2: 使用 API Key
swanlab login --api-key YOUR_API_KEY

# 方法 3: 环境变量
export SWANLAB_API_KEY="YOUR_API_KEY"
```

获取 API Key: https://swanlab.cn/settings/overview

### 步骤 3: 修改训练脚本

编辑 `sft_single_gpu_swanlab.sh`，设置模型路径：

```bash
BASE_MODEL="./models/Qwen2.5-0.5B-Instruct"  # 改为你的实际路径
```

### 步骤 4: 开始训练

```bash
chmod +x sft_single_gpu_swanlab.sh
bash sft_single_gpu_swanlab.sh
```

### 步骤 5: 查看结果

点击终端输出的链接，或访问：https://swanlab.cn/@your-username/MiniOneRec-0.5B

## 主要功能

### 1. 实时指标监控
- train/loss, eval/loss
- learning_rate
- GPU 显存和利用率

### 2. 超参数记录
自动记录所有训练参数

### 3. 实验对比
支持多个实验的可视化对比

### 4. 离线模式
```bash
export SWANLAB_MODE=offline
# 训练后同步: swanlab sync ./swanlog/run-xxxx
```

## 常见问题

**Q: 找不到实验记录？**
检查本地日志：`ls -la ./swanlog/`

**Q: 离线使用？**
设置 `SWANLAB_MODE=offline`，训练后使用 `swanlab sync` 同步

**Q: SwanLab vs WandB？**
SwanLab 国内访问快、完全免费、支持中文界面

## 资源

- 官网: https://swanlab.cn
- 文档: https://docs.swanlab.cn
- GitHub: https://github.com/SwanHubX/SwanLab
