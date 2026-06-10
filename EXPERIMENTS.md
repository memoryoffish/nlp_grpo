# Game24 实验日志 / 训练思路与参数全记录

> 目的：完整记录所有训练思路、超参、配置与结果，供后续写报告/PPT 引用。
> 任务：Qwen2.5-1.5B-Instruct 解 24 点，R1 风格 `<think>…</think><answer>…</answer>`，
> RLVR（程序判定奖励）。框架 veRL/TinyZero。配套：`REPORT.md`（成文报告）、
> `results/ablation_summary.md`（汇总表）、`results/{comparison,grpo_curves,…}`（原始数据）。

---

## 0. TL;DR（核心数字 + 结论）
- **难测试集（最难 100 题）可部署求解率：1%（贪心）→ ~50%（验证器 best-of-64, T=1.5, ±10pp）**。
- **分布内 greedy：27%（SFT）→ 42%（+sparse GRPO）**。
- **核心发现**：GRPO 是"利用"算子 → 把分布内做尖（27→42%），但**摧毁采样多样性** → 难测试 best-of-N 从 21% 塌到 5%。难题突破靠"探索"（强 SFT 多样性 + 验证器 best-of-N）。
- **缓解**：高 KL 锚定 + shaped 奖励 + 早停 → 分布内 31% 且 best-of-N(bv@16) 保住 18%。
- **最终管线（含 GRPO，按要求）**：`SFT v2 → 多样性保护型 GRPO → best-of-N 推理`。

---

## 1. 环境与基础设施（硬约束）
| 项 | 值 |
|---|---|
| venv | `/mnt/workspace/akide/code/unirlm-02/ddl_work/.venv311`（py3.11） |
| torch / cuda | 2.6.0+cu118 / 运行时 11.8 | 
| 其他 | flash-attn 2.6.3, ray 2.51.2, transformers 4.47.1 |
| GPU | 8× A100-80GB |
| 模型 | `/mnt/workspace/akide/models/Qwen2.5-1.5B-Instruct` |
| **vLLM** | **不可用**：驱动仅支持 CUDA 11.4（R470），现代 vLLM 需 CUDA 12.x（缺 `cuTensorMapEncodeTiled`），无 root 升不了 → **全程 HF rollout** |

**关键基础设施改造（verl 源码）**：
- `verl/workers/rollout/hf_rollout.py`：HF rollout 支持 `num_return_sequences=n`（GRPO 组采样必需，原版忽略 `rollout.n`），`do_sample=True` 时按 n 扩展 idx/attention_mask/position_ids/batch_size。
- `verl/utils/torch_functional.py`：`VERL_DISABLE_FLASH_ATTN_CE=1` 开关，回退 flash-attn CE → naive logsumexp（本环境 triton CE kernel 崩）。
- `verl/trainer/main_ppo.py`：`ray.init(runtime_env=...)` 传播 `VERL_DISABLE_FLASH_ATTN_CE/HF_ENDPOINT/HF_HOME/GAME24_REWARD/GAME24_SHAPING_COEF` 到 Ray worker（预启动 head 时 worker 不继承启动环境，否则 CE flag/奖励模式传不进去）。
- `verl/trainer/ppo/ray_trainer.py`：`+trainer.final_val=False` 开关（短跑省最终验证）。

**并发技巧（关键）**：多个 GRPO 同时跑用**隔离 Ray 集群**——每臂 `ray start --head` 独立端口/temp-dir + `SKIP_RAY_STOP=1` + `RAY_ADDRESS` 指向各自 head + `CUDA_VISIBLE_DEVICES` 限定卡（物理隔离，互不干扰）。脚本：`scripts/start_head.sh <gpus> <port> <reward>`。
**速度**：HF rollout 约 **70-90s/步**（2 卡/臂，batch16, resp320）；每臂 160 步 ~3-4h。**off-by-one**：`total_training_steps=N` 实跑 N-1 步；1 epoch=⌈1262/batch⌉ 步（batch16→78），故要 ≥3 epoch 才到 160 步。

---

## 2. 数据
**划分（同源按难度，`examples/data_preprocess/game24.py`）**：`test-time-compute/game-of-24`（1362 题）
- 训练：Rank 1–1262（最易，solved_rate 99.2%–61.7%）
- 测试：Rank 1263–1362（最难 100，solved_rate 58.0%–20.7%，均可解、与训练数字多重集 0 重叠）
- 幻觉：从 {1..13}⁴ 合成的无解四元组 100（精确求解器验证全无解）
- `reward_model.ground_truth` 含 `{numbers, target:24, solved_rate}`；prep 内置 train/test 不相交断言。
- prompt：qwen-instruct 模板，结尾预填 `<|im_start|>assistant\nLet me solve this step by step.\n<think>`；
  system 用 Qwen 默认（与 SFTDataset 套模板一致，保证 SFT→GRPO prompt 逐字对齐）。
- OOD（加分）：`Jiayi-Pan/Countdown-Tasks-3to4`，test 1024（评测取样 256）。

**SFT 数据（`examples/data_preprocess/generate_sft_game24.py`，solver 枚举全部解 + ToT 轨迹）**：
| 版本 | 每题解数 | 失败尝试 | 自检 | 规模 | 说明 |
|---|---|---|---|---|---|
| v1 | 1（最短） | 2（整数式） | 无 | 1199/63 | 初版 |
| **v2** | **2（最短 + 分数/除法解）** | **3（含 1 个分数尝试）** | **`Check: … = 24 ✓`** | **2391/125** | `--solutions_per_puzzle 2` |
- v2 诊断：贪心已能解一道难题 `(7-5)*9+6`，并产出分数式 `(7-10/4)*8`。response ~211 tok（<320 GRPO 上限）。

---

## 3. 奖励设计（`verl/utils/reward_score/game24.py`，env 切换）
| 模式 | 规则 |
|---|---|
| `sparse`（默认/对照锚） | 1.0 正确 / 0.1 有 `<answer>` 但数字或值错 / 0.0 无 answer |
| `shaped` | 数字对、值≠24 → `0.1 + 0.5·max(0,1-|v-24|/24)`（封顶 0.6）；**仅数字用对时**给（防 hack） |
- 验证器：取最后 `<answer>`，校验恰好用给定 4 数各一次，`safe_eval`（字符白名单 + **拒绝 `**`/`//`**），`|v-24|<1e-5`。
- env：`GAME24_REWARD=sparse|shaped`、`GAME24_SHAPING_COEF`（默认 0.5）。
- **被否决的想法**：难度加权 `reward*=1+(1-solved_rate)` —— 按题乘常数在 GRPO **组内优势归一化**下被约掉，**无效**。
  （唯有逐样本变化的 shaped 奖励能打破"全 0.1 零方差组"。）
- **分阶段奖励（staged）** = 两段顺序：Phase-1 `shaped` ~120 步 → Phase-2 从其 ckpt 换 `sparse` 再跑（去掉"接近"拐杖逼精确）。

---

## 4. SFT 训练配置（`scripts/train_sft_game24.sh`，fsdp_sft_trainer，torchrun）
| 参数 | v1 | v2 |
|---|---|---|
| lr | 2e-5 | **1e-5** |
| epochs | 3 | **4** |
| max_length | 1024 | 1024 |
| micro / train batch | 4 / 32 | 4 / 32 |
| 数据 | data/sft_game24 | data/sft_game24_v2 |
| 产出 | `game24-sft/global_step_111` | `game24-sft-v2/global_step_296` |
| 收敛 | train loss ~0.025 | val loss ~0.028 |

---

## 5. GRPO 训练配置（`scripts/train_game24_grpo_hf.sh`，HF rollout）
**共用基线**：`adv_estimator=grpo`, `rollout.name=hf`, `+rollout.micro_batch_size=1`, `top_k=0`,
`use_kl_loss=True kl_loss_type=low_var_kl`, `use_remove_padding=False`, fsdp offload=False,
`train_batch_size=16 ppo_mini=16 ppo_micro=4 ref_micro=4`, `max_prompt=256 max_resp=320`,
`test_freq=20`, `VERL_DISABLE_FLASH_ATTN_CE=1`。可调 env：`ROLLOUT_N LR KL_COEF ENTROPY_COEFF TEMPERATURE TOTAL_EPOCHS TOTAL_STEPS SAVE_FREQ N_GPUS GAME24_REWARD`。

**所有 GRPO 实验臂**：
| 臂 | init | reward | n | lr | KL | ent | temp | 步数 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| 纯GRPO(phase1) | base | sparse | 4 | 1e-6 | .001 | .001 | 1.0 | 160 | 对照：冷启动直接 GRPO |
| SFT→GRPO(phase1) | SFT v1 | sparse | 4 | 1e-6 | .001 | .001 | 1.0 | 160 | 主线对比 |
| v2-control | SFT v2 | sparse | 4 | 1e-6 | .001 | .001 | 1.0 | →80 | Wave-1 锚 |
| v2-n8 | SFT v2 | sparse | 8 | 1e-6 | .001 | .001 | 1.0 | →80 | n 消融 |
| v2-shaped | SFT v2 | shaped | 4 | 1e-6 | .001 | .001 | 1.0 | →80 | 奖励塑形 |
| v2-staged-p1 | SFT v2 | shaped | 4 | 1e-6 | .001 | .001 | 1.0 | →80 | 分阶段第一段 |
| div A | SFT v2 | shaped | 4 | 1e-6 | **.005** | **.01** | **1.2** | →40 | 多样性保护(低KL) |
| div B | SFT v2 | shaped | 8 | 1e-6 | **.01** | **.02** | **1.2** | →40 | 中等 |
| div C | SFT v2 | shaped | 4 | 1e-6 | **.03** | .01 | 1.2 | →20 | **最强 KL 锚定** |
| div D | SFT v2 | shaped | 4 | 1e-6 | .005 | **.03** | **1.3** | →20 | 最强探索 |

---

## 6. 评测方法（`scripts/eval_compare.py`，全程 GAME24_REWARD=sparse，seed=0）
- **greedy**（do_sample=False，1-shot）+ **best_verified@k**（采 k 个、取首个验证器判正确的 → 可部署求解率，因有精确验证器 = pass@k）+ **maj@k**（多数投票）。frontier k∈{1,2,4,8,16,32,64}。
- 温度 sweep（best-of-N 用）：T∈{1.0,1.2,1.5}。
- 数据集：分布内(train 取样100) / 难测试(100) / 幻觉(100) / Countdown OOD(256)。
- 快筛模式：`--train_n 0 --countdown_n 0 --no_halluc --bon 8`。
- 曲线：`scripts/analyze_grpo.py` 解析训练 log → `results/grpo_curves.{csv,png}`。

---

## 7. 关键结果

### 7.1 第一阶段：SFT→GRPO vs 纯 GRPO（SFT v1，`results/comparison.md`）
| 模型 | 分布内 | 难测试贪心 | 难测试 pass@4 | 幻觉↓ | Countdown OOD |
|---|---|---|---|---|---|
| base | 0% | 0% | 0% | 0% | 1.95% |
| 纯 GRPO | 5% | 1% | 0% | 0% | 2.34% |
| SFT-only(v1) | 20% | 1% | 3% | 0% | 5.08% |
| SFT→GRPO(v1) | 27% | 1% | 5% | 0% | 12.89% |
> 结论：受控对比下 SFT→GRPO 全面优于纯 GRPO（分布内 5.4×、OOD 5.5×）；纯 GRPO 受"零梯度组"制约。

### 7.2 第二阶段：best-of-N 天花板（SFT v2，`results/ablation_summary.md`）
| 温度 | greedy | bv@8 | bv@16 | bv@32 | **bv@64** |
|---|---|---|---|---|---|
| 1.0 | 3% | 12% | 16% | 26% | 37% |
| 1.5 | 0% | 8% | 17% | 38% | **50%** |

### 7.3 探索/利用权衡（seeded）
| 模型 | 分布内 | bv@4 | bv@8 | bv@16 |
|---|---|---|---|---|
| SFT v2(无GRPO) | 27% | 7% | 13% | **21%** |
| sparse GRPO@40 | 27% | 4% | 8% | 12% |
| sparse GRPO@80 | **42%** | 2% | 2% | 5% |
| div 高KL@20 | 31% | **11%** | 12% | 18% |
| div 低KL@20 | 31% | 7% | **14%** | 18% |
| div 高熵@20 | 26% | 3% | 7% | 13% |

### 7.4 定性（`results/qualitative_samples.md`）
纯 GRPO（base 初始化）常**照抄 prompt 示例** `(1+2)*8`、无视题目数字；SFT→GRPO 用正确数字真实尝试并常算对。

---

## 8. 可信度（对抗式验证 workflow，`verify-game24-findings`）
- 50% 天花板、GRPO 坍缩两结论均判 **SOUND**；验证器无假阳性、train/test 0 重叠、无截断伪影。
- **caveat**：50% = 单次 best-of-64 抽样(n=100)，写 **~50%±10pp**；"可部署"需 **64× 推理 + 部署时验证器**；坍缩"端点显著"(z=3.21)，不宣称逐步单调。
- 修复：验证器拒 `**`/`//`（原 0 次触发，数值不变）；eval 加种子；归因统一到 control 臂。

---

## 9. 复现实验（命令）
```bash
cd ddl_work/project/TinyZero
# 数据
bash scripts/prepare_data_game24.sh          # game24 (train1262/test100/halluc100)
bash scripts/prepare_data_countdown.sh       # OOD
INPUT=data/game24/train.parquet OUTPUT_DIR=data/sft_game24_v2 \
  .venv311/bin/python examples/data_preprocess/generate_sft_game24.py \
  --input_parquet data/game24/train.parquet --output_dir data/sft_game24_v2 --solutions_per_puzzle 2
# SFT v2
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS=4 SFT_DATA_DIR=$PWD/data/sft_game24_v2 \
  SFT_SAVE_DIR=$PWD/checkpoints/TinyZero/game24-sft-v2 LR=1e-5 TOTAL_EPOCHS=4 \
  bash scripts/train_sft_game24.sh
# GRPO（隔离 Ray 并发：先 start_head 再带 RAY_ADDRESS+SKIP_RAY_STOP=1 起臂）
bash scripts/start_head.sh 0,1 6379 shaped
RAY_ADDRESS=<ip>:6379 SKIP_RAY_STOP=1 CUDA_VISIBLE_DEVICES=0,1 N_GPUS=2 \
  GAME24_REWARD=shaped ROLLOUT_N=4 KL_COEF=0.03 ENTROPY_COEFF=0.01 TEMPERATURE=1.2 \
  BASE_MODEL=$PWD/checkpoints/TinyZero/game24-sft-v2/global_step_296 \
  TOTAL_EPOCHS=2 TOTAL_STEPS=80 SAVE_FREQ=20 EXPERIMENT_NAME=game24-grpo-v2-divC \
  bash scripts/train_game24_grpo_hf.sh
# 评测（best-of-N + 温度）
.venv311/bin/python scripts/eval_compare.py \
  --model "sftv2:checkpoints/TinyZero/game24-sft-v2/global_step_296" \
  --bon 64 --temperature 1.5 --no_halluc --countdown_n 0 --seed 0
```

## 10. 文件 / checkpoint 索引
- 报告：`REPORT.md`（成文，含第 7 章提升）；本文件 `EXPERIMENTS.md`（参数全记录）。
- 结果：`results/comparison.{md,json}`、`results/ablation_summary.md`、`results/qualitative_samples.md`、`results/{screen,ceiling,bv16,divsweep}/`。
- 训练曲线（REPORT §8）：`results/curves_phase1.png`（纯GRPO vs SFT→GRPO 奖励/验证分）、`results/curves_wave1.png`（奖励设计消融）、`results/curves_entropy.png`（熵曲线=多样性坍缩机制）；旧 `results/grpo_curves.png`。
- 监控：训练用 console logger 逐步打印（`game24-grpo-*.log`），`WANDB_MODE=offline`（离线 run 在 `wandb/`，3 个早期 run）；曲线由 `scripts/analyze_grpo.py` 解析日志生成，不依赖 wandb 在线面板。
- checkpoints（`checkpoints/TinyZero/`）：`game24-sft`(v1 111)、`game24-sft-v2`(296)、`game24-grpo-base/sftinit`(40/80/120, phase1)、`game24-grpo-v2-{control,n8,shaped,staged-p1}`(40/80)、`game24-grpo-v2-div{A,B,C,D}`(20/40)。
- 改动文件：`examples/data_preprocess/{game24,generate_sft_game24}.py`、`verl/utils/reward_score/game24.py`、`verl/workers/rollout/hf_rollout.py`、`verl/utils/torch_functional.py`、`verl/trainer/{main_ppo,ppo/ray_trainer}.py`、`scripts/{train_sft_game24,train_game24_grpo_hf,eval_compare,analyze_grpo,start_head}.{sh,py}`。
