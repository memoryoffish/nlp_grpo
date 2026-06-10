# TinyZero — Game of 24 GRPO 训练指南

基于 [veRL](https://github.com/volcengine/verl) 框架，使用 GRPO（Group Relative Policy Optimization）算法对 Qwen2.5-1.5B-Instruct 进行强化学习训练，任务为 24 点游戏。

---

## 目录

1. [环境要求](#环境要求)
2. [环境配置](#环境配置)
3. [模型准备](#模型准备)
4. [数据准备](#数据准备)
5. [冒烟测试](#冒烟测试)
6. [SFT 冷启动（推荐）](#sft-冷启动推荐)
7. [启动训练](#启动训练)
8. [监控训练（wandb）](#监控训练wandb)
9. [Checkpoint 说明](#checkpoint-说明)
10. [加分项：Countdown OOD 泛化验证](#加分项countdown-ood-泛化验证)

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（Ubuntu 20.04+） |
| Python | 3.9 |
| CUDA | 12.1 (其他版本都没跑通) |
| GPU 显存 | 单卡至少 20 GB（训练 1.5B 模型） |
| conda | Miniconda 或 Anaconda |

---

## 环境配置

```bash
# 1. 创建 conda 环境
conda create -n zero python=3.9 -y
conda activate zero

# 2. 安装 PyTorch（其他版本都没跑通）
# CUDA 12.1：
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# 3. 安装 vLLM（推理引擎）
pip install vllm==0.6.3

# 4. 安装 Ray（分布式调度）
pip install ray==2.51.2

# 5. 安装项目本身（verl）
cd /path/to/TinyZero
pip install -e .

# 6. 安装 Flash Attention 2（加速注意力计算，耗时较长）
pip install flash-attn==2.8.3 --no-build-isolation

# 7. 安装其他依赖
pip install wandb IPython matplotlib
```


---

## 模型准备

从 Hugging Face 下载模型到本地（或手动拷贝）：

```bash
# 方式一：使用 huggingface-cli（推荐）
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct \
    --local-dir models/Qwen2.5-1.5B-Instruct

# 方式二：使用 modelscope（国内网络）
pip install modelscope
modelscope download --model Qwen/Qwen2.5-1.5B-Instruct \
    --local_dir models/Qwen2.5-1.5B-Instruct
```

下载后目录结构应如下：

```
TinyZero/
└── models/
    └── Qwen2.5-1.5B-Instruct/
        ├── config.json
        ├── tokenizer.json
        ├── model.safetensors（或分片文件）
        └── ...
```

---

## 数据准备

数据来源于 HuggingFace 的 `test-time-compute/game-of-24` 数据集（1362 道有解谜题），按难度自动划分为训练集和测试集，**无需人工标注**。

```bash
# 从项目根目录运行
conda activate zero
bash scripts/prepare_data_game24.sh
```

生成结果保存在 `./data/game24/`：

| 文件 | 样本数 | 说明 |
|---|---|---|
| `train.parquet` | 1262 | Rank 1–1262（最简单，训练用） |
| `test.parquet` | 100 | Rank 1263–1362（最难，OOD 测试） |
| `hallucination.parquet` | 100 | 合成无解四元组（幻觉测试） |

如需自定义大小：

```bash
TRAIN_SIZE=500 TEST_SIZE=50 bash scripts/prepare_data_game24.sh
```

---

## 冒烟测试

正式训练前，先运行冒烟测试验证整条链路（数据预处理 → 奖励函数 → 训练 1 个 epoch）：

```bash
conda activate zero
bash scripts/smoke_test_game24.sh
```

测试包含三步：
1. 生成微型数据集（32 训练 / 16 测试）
2. 验证奖励函数正确性
3. 跑 1 个 epoch 的训练

看到以下输出说明环境正常：

```
Smoke test PASSED — pipeline is working.
```

---

## SFT 冷启动（推荐）

> **为什么需要 SFT？** 基模在难题集（Rank 1263–1362）上初始解题率约 1%，GRPO 以 n=4 采样时，每组 4 条输出里至少有 1 条正确的概率只有 ~4%，96% 的组梯度为零，训练很难收敛。SFT 冷启动先教模型"如何搜索"，使解题率提升到 20–40%，再用 GRPO 精细化。

### 流程

```
prepare_data_game24.sh   →  data/game24/train.parquet
         ↓
generate_sft_data_game24.sh  →  data/sft_game24/{train,val}.parquet
         ↓
train_sft_game24.sh      →  checkpoints/TinyZero/game24-sft/
         ↓
train_game24_local.sh    →  BASE_MODEL 指向 SFT checkpoint，跑 GRPO
```

### 第一步：生成 SFT 数据

程序对每道训练题调用精确 solver 枚举所有解，再生成包含搜索过程的推理轨迹：

```bash
# 先确保 game24 GRPO 数据已生成
bash scripts/prepare_data_game24.sh

# 生成 SFT 轨迹数据
bash scripts/generate_sft_data_game24.sh
# 输出：data/sft_game24/train.parquet（~1198 条）和 val.parquet（~64 条）
```

生成的每条样本格式：

```
PROMPT: Using the numbers [4, 8, 11, 11], create an equation that equals 24 ...

RESPONSE:
Let me solve this step by step.
<think>
I need to make 24 from [4, 8, 11, 11], using each number exactly once.

Let me try some combinations first:
- Try 11 + 11 + 8 - 4 = 26 ≠ 24. Does not work.
- Try 11 * 8 - 4 - 11 = 73 ≠ 24. Does not work.

Let me think differently:
8 / 4 + 11 + 11 = 24 ✓  Found it!
</think>
<answer>8 / 4 + 11 + 11</answer>
```

### 第二步：SFT 训练

```bash
conda activate zero
bash scripts/train_sft_game24.sh
# 约 3 epochs，单卡约 1–2 小时
# checkpoint 保存在 checkpoints/TinyZero/game24-sft/
```

### 第三步：用 SFT checkpoint 初始化 GRPO

```bash
# 找到最新的 SFT checkpoint
SFT_CKPT=$(ls -td checkpoints/TinyZero/game24-sft/global_step_* | head -1)

# 以 SFT checkpoint 为起点运行 GRPO
BASE_MODEL=$SFT_CKPT bash scripts/train_game24_local.sh
```

---

## 启动训练

所有脚本需从**项目根目录**运行，模型和数据路径默认使用 `$PWD` 相对路径。

### 单卡训练（适合 1.5B 模型，显存 ≥ 20 GB）

```bash
conda activate zero
bash scripts/train_game24_local.sh
```

主要配置（[scripts/train_game24_local.sh](scripts/train_game24_local.sh)）：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `BASE_MODEL` | `$PWD/models/Qwen2.5-1.5B-Instruct` | 模型路径 |
| `DATA_DIR` | `$PWD/data/game24` | 数据路径 |
| `train_batch_size` | 32 | 每步训练样本数 |
| `rollout.n` | 4 | 每道题采样输出数 |
| `total_epochs` | 15 | 训练总轮数 |
| `save_freq` | 50 | 每隔多少步保存 checkpoint |

### 多卡训练（适合更大 batch 或更大模型）

```bash
export N_GPUS=4
export ROLLOUT_TP_SIZE=4   # vLLM 张量并行度，通常等于 N_GPUS
bash scripts/train_game24.sh
```


---

## 监控训练（wandb）

### 登录 wandb

```bash
wandb login   # 粘贴 API key（https://wandb.ai/authorize）
```

### 指定个人 workspace（避免同步到 Team）

```bash
export WANDB_ENTITY=你的wandb用户名   # 登录时显示的用户名
```

完全离线（不同步）：

```bash
export WANDB_MODE=offline
# 训练结束后手动上传：wandb sync wandb/latest-run
```

### 关键指标说明

| 指标 | 含义 | 预期趋势 |
|---|---|---|
| `critic/score/mean` | 每步平均奖励（核心指标） | 从 ~0.09 逐渐升高 |
| `critic/score/max` | 批次内最高奖励 | 应尽早出现 1.0 |
| `val/test_score/game24` | OOD 测试集得分 | 随训练上升，衡量泛化 |
| `actor/kl_loss` | KL 散度 | 保持较小值（< 0.5） |
| `response_length/mean` | 平均输出长度 | 若持续增长说明模型在"思考" |
| `actor/entropy_loss` | 策略熵 | 不应过快下降（防止过早收敛） |

---

## Checkpoint 说明

Checkpoint 自动保存在：

```
checkpoints/TinyZero/<实验名>/actor/global_step_<N>/
```

例如使用默认配置时：

```
checkpoints/TinyZero/game24-qwen2.5-1.5b-grpo-local/
└── actor/
    ├── global_step_50/
    ├── global_step_100/
    └── ...
```

默认每 **50 步**保存一次（`save_freq=50`），训练前 50 步目录不存在是正常的。

加载 checkpoint 推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "checkpoints/TinyZero/game24-qwen2.5-1.5b-grpo-local/actor/global_step_50"
)
```

---

## 模型评估

训练过程中验证集得分会实时上报 wandb。训练结束后可用以下脚本对任意 checkpoint 做离线评估。

### In-distribution 评估（game24 测试集 + 幻觉集）

```bash
# 评估 step 200 checkpoint 在 game24 测试集上的准确率
STEP=200 bash scripts/eval_game24.sh

# 同时评估幻觉测试集（无解题，检查模型是否会瞎编算式）
STEP=200 HALLUC=1 bash scripts/eval_game24.sh
```

输出示例：
```
game24    : accuracy=62.00%  (62/100)
hallucination: accuracy=3.00%  (3/100)   ← 越低越好，说明模型不乱编
```

### OOD 评估（countdown 测试集，加分项）

```bash
# 先准备 countdown 数据（一次性）
bash scripts/prepare_data_countdown.sh

# 评估 game24 模型在 countdown 上的泛化能力
STEP=200 bash scripts/eval_ood.sh

# 同时对比 in-distribution 和 OOD
STEP=200 EXTRA_DATA=$PWD/data/game24/test.parquet bash scripts/eval_ood.sh
```

### 批量评估多个 checkpoint

```bash
for STEP in 50 100 150 200 250 300; do
    STEP=$STEP HALLUC=1 bash scripts/eval_game24.sh
    STEP=$STEP bash scripts/eval_ood.sh
done
```

结果保存在 `results/` 目录下，每个 step 一个 JSON 文件，可用于绘制准确率随训练步数变化的曲线。

---

## 奖励函数说明

奖励函数位于 [verl/utils/reward_score/game24.py](verl/utils/reward_score/game24.py)：

| 情况 | 奖励 |
|---|---|
| 使用恰好给定的 4 个数字，表达式结果等于 24 | **1.0** |
| 有 `<answer>` 标签但数字或结果不对 | **0.1** |
| 没有 `<answer>` 标签 | **0.0** |

---

## 加分项：Countdown OOD 泛化验证

> 对应课程要求：`Jiayi-Pan/Countdown-Tasks-3to4`——3–4 数字凑任意目标数，用于 OOD 验证与任务扩展。

**思路**：在 Game24（固定目标 = 24，数字范围 1–13）上训练，然后在 Countdown（任意目标 1–1000，数字范围 1–100）测试集上评估，验证模型学到的是通用算术推理能力，而非对 24 的记忆。

### 第一步：准备 Countdown 测试数据

```bash
bash scripts/prepare_data_countdown.sh
# 生成 data/countdown/train.parquet（10000条）和 test.parquet（1024条）
# OOD 评估只用到 test.parquet
```

### 第二步：正常训练 Game24 模型

```bash
bash scripts/train_game24_local.sh
# checkpoint 保存在 checkpoints/TinyZero/game24-qwen2.5-1.5b-grpo-local/actor/
```

### 第三步：OOD 评估

```bash
# 评估某个 checkpoint 在 countdown 测试集上的得分
STEP=200 bash scripts/eval_ood.sh

# 同时评估 game24 测试集（对比 in-distribution vs OOD）
STEP=200 EXTRA_DATA=$PWD/data/game24/test.parquet bash scripts/eval_ood.sh
```

结果保存在 `results/ood_eval_step200.json`，格式如下：

```json
{
  "model": "checkpoints/.../global_step_200",
  "results": {
    "countdown": {
      "n_samples": 1024,
      "mean_score": 0.21,
      "accuracy": 0.18,
      "n_correct": 184
    },
    "game24": {
      "n_samples": 100,
      "mean_score": 0.65,
      "accuracy": 0.62,
      "n_correct": 62
    }
  }
}
```

也可以对多个 checkpoint 批量评估，追踪 OOD 准确率随训练步数的变化：

```bash
for STEP in 50 100 150 200; do
    STEP=$STEP EXTRA_DATA=$PWD/data/game24/test.parquet bash scripts/eval_ood.sh
done
```

### 关键脚本说明

| 脚本 | 说明 |
|---|---|
| [scripts/prepare_data_countdown.sh](scripts/prepare_data_countdown.sh) | 下载并处理 Countdown 数据集 |
| [examples/eval_ood.py](examples/eval_ood.py) | 核心评估脚本，支持任意 parquet 测试集 |
| [scripts/eval_ood.sh](scripts/eval_ood.sh) | 封装 eval_ood.py，一行命令运行 |
| [verl/utils/reward_score/countdown.py](verl/utils/reward_score/countdown.py) | Countdown 奖励函数（任意目标数） |
