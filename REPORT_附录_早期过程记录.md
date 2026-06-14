# 附录：早期过程记录（原 REPORT §1–§8）

> ⚠️ **本文件全部内容不合规且含数据泄漏（测试集与训练集重叠），所有绝对数字与结论均已作废，请勿引用。**
> 保留仅为展示研究演进、监控与调参过程（对应评分点「必要时调超参」）。
> **权威结果见 [`REPORT.md`](REPORT.md) §9、[`补充材料.md`](补充材料.md) §10。**

---

## 1. 动机：纯 GRPO 的"零梯度组"问题

GRPO 用同一道题的 `n` 个采样输出组成一个 group，用组内相对奖励做优势估计。
若一道题的 `n` 个输出**全部**得到相同奖励（例如全 0），该 group 的优势恒为 0，**不产生梯度**。

基座模型在难题集上几乎不会输出合法格式 / 正确算式，于是绝大多数 group 全 0 → 训练信号极稀疏。
本实验**实测**证实了这一点：

- 基座模型在难测试集（最难 100 题）上的初始验证分 **val/test_score = 0.061**
  （奖励 1.0/0.1/0.0；0.061 意味着大部分输出连合法 `<answer>` 都没有，只有零星 0.1）。

SFT 冷启动的作用：先用**带搜索过程的推理轨迹**教会模型"如何尝试-验证-给出答案"，
使其**稳定输出 R1 格式**且**经常能采样到正确解**，从而让后续 GRPO 的 group 内出现非零奖励差异、
产生有效梯度。

---

## 2. 方法

### 2.1 SFT 冷启动数据（Arm B 第一阶段）
对每道训练题用**精确求解器**（`solve24`，分数运算枚举所有解）找出全部正确算式，
再程序化生成一段 **Tree-of-Thought 风格**的推理轨迹：先给出 2 个"尝试-失败"的组合，
再给出正确解。示例：

```
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
- 规模：1199 训练 / 63 验证（每题取最简解）。轨迹完全确定式生成，无 LLM 噪声。
- SFT：`fsdp_sft_trainer`，lr 2e-5，3 epochs，max_length 1024，train loss 收敛到 ~0.025。

### 2.2 GRPO（两个 Arm 共用）
- 算法：GRPO（`adv_estimator=grpo`），组内相对优势，`rollout.n=4`。
- 奖励 `verl/utils/reward_score/game24.py`：
  - **1.0**：恰好用给定 4 数、结果 = 24；
  - **0.1**：有 `<answer>` 但数字或结果不对（格式分）；
  - **0.0**：无 `<answer>`。
- KL 正则 `kl_loss_coef=0.001`（low_var_kl），entropy_coeff=0.001。

### 2.3 提示与 SFT/GRPO 格式对齐（关键）
GRPO/评测使用 `qwen-instruct` 提示，结尾预填 `<|im_start|>assistant\nLet me solve this step by step.\n<think>`。
SFT 数据的 prompt 为纯问题文本，由 `SFTDataset` 套用 chat template。
**为保证 SFT 暖启动能干净迁移到 GRPO**，我们把 GRPO 提示的 system 改为 Qwen 默认
（"You are Qwen, created by Alibaba Cloud. You are a helpful assistant."），
使 `GRPO prompt == SFT 渲染 prompt + "<think>" 预填`（已逐字符校验一致，仅预填部分不同，
而预填正是 SFT 训练模型学会续写的内容）。两个 Arm 用**完全相同**的提示。

---

## 3. 实验设置

### 3.1 数据划分（与参考实现 (3) 一致，同源按难度切分）
| 划分 | 来源 | 数量 | 说明 |
|---|---|---|---|
| 训练 | test-time-compute/game-of-24 Rank 1–1262 | 1262 | 最易（solved_rate 99.2%–61.7%） |
| 测试 | 同数据集 Rank 1263–1362 | 100 | 最难（solved_rate 58.0%–20.7%），衡量泛化 |
| 幻觉集 | 从 {1..13}⁴ 合成的无解四元组 | 100 | 检测模型是否对无解题"瞎编"算式 |
| OOD（加分） | Jiayi-Pan/Countdown-Tasks-3to4 | 256（取样） | 3–4 数凑任意目标，验证通用算术推理 |

### 3.2 运行环境
- 仅 `uv` (Python 3.11) 环境可用：torch 2.6.0+cu118、flash-attn 2.6.3、ray 2.51.2。
- 该环境内 vLLM ABI 损坏（`undefined symbol: cuTensorMapEncodeTiled`，cu12.x vs cu118），
  且无 conda `zero` 环境，故 rollout 后端使用 **HF rollout**（而非参考脚本的 vLLM）。
- 为支持 HF rollout 下 GRPO 的 `n>1` 组采样，对 `hf_rollout.py` 增加 `num_return_sequences=n`
  并相应扩展张量；另加 `VERL_DISABLE_FLASH_ATTN_CE` 开关规避 flash-attn CE 兼容问题。

### 3.3 受控对比配置（两个 Arm 完全一致，仅 BASE_MODEL 不同）
`rollout.n=4`, `train_batch_size=16`, `max_response_length=256`, `lr=1e-6`,
`temperature=1.0`, `total_training_steps=160`, `test_freq=20`, `save_freq=40`, 4×A100。
- **Arm A（纯 GRPO）**：BASE_MODEL = Qwen2.5-1.5B-Instruct。
- **Arm B（SFT→GRPO）**：BASE_MODEL = SFT checkpoint（`game24-sft/global_step_111`）。

### 3.4 设置正确性审计（已通过）
为保证对比有效，训练前对设置做了对抗式审计：
- **无数据泄漏**：训练(1262) ∩ 测试(100) = 0；幻觉集与二者均不相交。
- **幻觉集确为无解**：用精确求解器验证 100 题**全部无解**（0/100 可解）→ 幻觉率指标有效。
- **测试集全部可解**：100 题均有解（对模型公平）。
- **奖励函数正确**：正确算式→1.0；数字/结果错→0.1；无 `<answer>`→0.0；裸 "24"→0.1（不被误判为正确）。

---

## 4. 结果

### 4.1 训练曲线（reward / 验证分 vs step，见 `results/grpo_curves.png`）
两条 arm 各跑 160 步（HF rollout，4×A100，曲线数据见 `results/grpo_curves.csv`）：

- **训练奖励 `critic/score/mean`（核心信号）**：
  - 纯 GRPO 全程卡在 **0.10–0.13**（≈ 仅 3–5% rollout 得到 1.0，其余只拿格式分 0.1），
    仅在最后 ~30 步出现微弱上扬到 ~0.20。
  - SFT→GRPO **从第 1 步起就在 0.25–0.44**（≈ 20–30% rollout 正确），全程是纯 GRPO 的 2–3 倍。
  - 即"SFT 让 GRPO 的组内一开始就有正确样本可强化"，而纯 GRPO 长期处于零梯度组主导的停滞区。
- **验证分 `val/test_score`（难测试集贪心）**：两条 arm 都长期在 **0.10** 附近
  （难测试集是最难的 100 题，1.5B 贪心 1-shot 接近上限，见 4.2 与局限性）。

### 4.2 最终对比表（`results/comparison.md`，checkpoint = global_step_120）
评测维度：分布内训练集采样 100 题贪心准确率、难测试集贪心 + pass@4、幻觉集"求解率"（越低越好）、
Countdown OOD（3–4 数任意目标，256 题）贪心准确率。

| Model | 分布内(train) 贪心 | 难测试 贪心 | 难测试 pass@4 | 难测试 mean | 幻觉求解率↓ | Countdown OOD |
|---|---|---|---|---|---|---|
| base (Qwen2.5-1.5B-Instruct) | 0% | 0% | 0% | 0.091 | 0% | 1.95% |
| 纯 GRPO (Arm A) | 5% | 1% | 0% | 0.109 | 0% | 2.34% |
| SFT-only（仅冷启动） | 20% | 1% | 3% | 0.109 | 0% | 5.08% |
| **SFT→GRPO (Arm B)** | **27%** | 1% | **5%** | 0.109 | 0% | **12.89%** |

### 4.3 定性样例（`results/qualitative_samples.md`）
训练中 reward 函数打印的 rollout 样例清楚显示两种 arm 的行为差异：

- **纯 GRPO（base 初始化）**：常常**照抄 prompt 里的少样本示例** `(1 + 2) * 8`，无视题目给的数字，
  例如题目 `[5,7,11,13]` 却输出 `(1 + 2) * 8 = 3 * 8 = 24`（数字错→0.1），几乎不真正求解。
- **SFT→GRPO**：使用**正确的数字**做真实尝试并经常正确，如
  `[2,2,7,13]→13 + 7 + 2 + 2`✓、`[4,4,4,8]→8 * 4 - 4 - 4`✓。

---

## 5. 分析（方法优越性与局限性）

**方法优越性（SFT→GRPO vs 纯 GRPO，受控对比）**
1. **分布内求解能力**：SFT→GRPO **27%** vs 纯 GRPO **5%**（**5.4×**）。纯 GRPO 训练 120 步后
   甚至**低于仅 SFT（20%）**，说明从冷启动直接 GRPO 很难学会真正求解。
2. **OOD 泛化（Countdown，完全 held-out）**：SFT→GRPO **12.89%** vs 纯 GRPO **2.34%**（**5.5×**）；
   且相对 SFT-only 的 5.08% 再翻 **2.5×**。这是最有力的证据：**GRPO 叠加在 SFT 之上学到的是可迁移的
   通用算术推理**（迁移到 3–4 数、任意目标的不同任务），而非对 24 的记忆。
3. **难测试 pass@4**：SFT→GRPO 5% vs 纯 GRPO 0%（贪心都≈0–1%，但有效采样能力 SFT→GRPO 更强）。
4. **GRPO 在 SFT 之上确有增益**：分布内 20%→27%、OOD 5.08%→12.89%，证明两阶段是互补的而非冗余。
5. **不产生幻觉**：所有模型在无解题上的"求解率"均为 0%，SFT/GRPO 没有诱发对无解题的瞎编。

**机制解释**：base 在难/新题上几乎不输出合法解，GRPO 以 n=4 采样时一组内全 0 → 优势为 0 → 无梯度
（零梯度组问题，实测纯 GRPO 训练奖励长期 ~0.1）。SFT 冷启动把"按格式输出 + 真实尝试"的能力先灌入，
使组内经常出现正确样本（训练奖励 0.25–0.44），GRPO 才能有效放大。

**局限性**
- **难测试集贪心接近天花板**：最难 100 题（人类解出率 20–58%）对 1.5B 贪心 1-shot 极难，两 arm 都 ≈0–1%，
  因此该指标不区分；区分度体现在分布内、pass@k 与 OOD。
- **算力/规模受限**：本机 vLLM 不可用，改用较慢的 HF rollout，故规模较小（160 步、batch 16、resp 256）。
  更长训练或可用 vLLM 提速后，纯 GRPO 末段出现的微弱上扬或可进一步发展，SFT→GRPO 也可能继续提升。
- **checkpoint**：因步数上限的 off-by-one，最终保存到 global_step_120（非 160）；曲线显示该处已接近稳定。
- **奖励较稀疏（1.0/0.1/0.0）**：对"接近但不等于 24"无塑形，主要靠 SFT 提供稠密起点。

**结论**：在严格受控（同数据/奖励/超参/提示，仅初始权重不同）的对比下，
**先 SFT 后 GRPO 在分布内求解（5.4×）与 OOD 泛化（5.5×）上显著优于纯 GRPO**，
且 GRPO 在 SFT 之上带来额外且可迁移的提升；纯 GRPO 从冷启动受零梯度组问题制约，长期停滞。

---

## 6. 复现实验
```bash
cd ddl_work/project/TinyZero
bash scripts/prepare_data_game24.sh           # 数据
bash scripts/prepare_data_countdown.sh        # OOD 数据
bash scripts/generate_sft_data_game24.sh      # SFT 轨迹
bash scripts/train_sft_game24.sh              # SFT 冷启动
# Arm A 纯 GRPO
EXPERIMENT_NAME=game24-grpo-base bash scripts/train_game24_grpo_hf.sh
# Arm B SFT→GRPO
BASE_MODEL=$(ls -td checkpoints/TinyZero/game24-sft/global_step_*|head -1) \
  EXPERIMENT_NAME=game24-grpo-sftinit bash scripts/train_game24_grpo_hf.sh
# 评测与对比
python scripts/eval_compare.py --model base:... --model SFT-only:... \
  --model pure-GRPO:... --model SFT-GRPO:... --out_md results/comparison.md
python scripts/analyze_grpo.py --log game24-grpo-base.log:pure-GRPO \
  --log game24-grpo-sftinit.log:SFT->GRPO --out_png results/grpo_curves.png
```

---

# 7. 进一步提升与消融（"刷上去" + 调参）

第二阶段在第一阶段(确认 SFT→GRPO 优于纯 GRPO)之上系统地把成绩往上刷。结论先行：
**最难测试集的可部署求解率从 1% 提升到 ~50%**，并发现了一个关键的**探索/利用权衡**。
所有数字均经独立对抗式验证(见 7.5),验证器已修复 `**`/`//` 漏洞(原结果 0 次触发,数值不变)。

## 7.1 三个抓手
1. **更强 SFT v2**(同 1262 训练题,**不扩充**避免泄漏):每题 2 个解(含分数/除法解)、3 次失败尝试(含 1 个分数尝试)、加一行自检 `Check: … = 24 ✓`;lr 1e-5、4 epochs。`examples/data_preprocess/generate_sft_game24.py`。
2. **验证器引导 best-of-N**(最便宜的大头):有精确验证器(算式逐字判定=24),推理时采样 N 个、输出验证为正确的那个。`scripts/eval_compare.py` 的 `best_verified@k`。
3. **奖励工程 + GRPO 超参消融**:`sparse` / `shaped`(`0.1+0.5·接近度`,仅数字用对时给,防 hack)经 `GAME24_REWARD` 切换;扫 n、lr、KL、熵、温度。
   > 注:**难度加权奖励被砍掉**——按题乘常数在 GRPO 组内优势归一化下会被约掉,是无效操作。

## 7.2 头条结果:难测试集求解率 1% → 50%
`SFT v2` 在最难 100 题上,验证器引导 best-of-N(单次抽样,n=100,~±10pp):

| 采样温度 | greedy | bv@8 | bv@16 | bv@32 | **bv@64** |
|---|---|---|---|---|---|
| T=1.0 | 3% | 12% | 16% | 26% | 37% |
| T=1.5 | 0% | 8% | 17% | 38% | **50%** |

- **高温=更多样=验证器能挑到的正确解更多**:T=1.5 在大 N 下最佳。
- "可部署"的前提要写清:**64× 推理成本 + 部署时有精确验证器**选答案;贪心 1-shot 仍只有 0-3%(接近 1.5B 单次天花板)。

## 7.3 核心发现:GRPO 的"探索/利用"权衡(seeded 消融)
| 模型 | 分布内 greedy | bv@4 | bv@8 | bv@16 |
|---|---|---|---|---|
| **SFT v2(无 GRPO)** | 27% | 7% | 13% | **21%** |
| sparse GRPO @40 | 27% | 4% | 8% | 12% |
| sparse GRPO @80 | **42%** | 2% | 2% | 5% |
| div(高KL/shaped)@20 | 31% | **11%** | 12% | 18% |
| div(低KL/shaped)@20 | 31% | 7% | **14%** | 18% |
| div(高熵/T1.3)@20 | 26% | 3% | 7% | 13% |

- **GRPO 把策略尖锐化**:sparse GRPO 训到 80 步,分布内 greedy 27%→**42%**,但**摧毁采样多样性** → 难测试 best-of-N 从 21%(bv@16)塌到 **5%**。两个目标赢家相反。
- 统计:base→80步 bv@16 下降显著(两比例 z=3.21, p<0.01),被 bv@4/bv@8 交叉印证;中间步在 n=100 单抽下不显著,故表述为"端点显著",不宣称"每步单调"。
- **缓解 = 多样性保护型 GRPO**:**高 KL 锚定**(把策略拉住在多样的 SFT 附近)+ shaped 奖励 + **早停(~20 步)**,能做到分布内 27%→31% **同时**保住 best-of-N(bv@16 21%→18%,远好于 sparse 的 5%)。过度探索(高熵/高温)反而变差。

## 7.4 最终管线(含 GRPO)与推荐
**SFT v2 → 多样性保护型 GRPO(shaped + 高 KL + 早停) → best-of-N 推理**:
- **要分布内/单答案最强** → sparse GRPO(42%);
- **要难题可部署求解率最高** → SFT v2 + best-of-N(bv@64 T1.5 ≈ 50%),GRPO 宜轻(早停/高KL)以免塌多样性;
- **平衡** → 多样性保护型 GRPO @~20 步:分布内 31% + bv@16 18%。

## 7.5 可信度(独立对抗式验证)
对头条结论做了对抗审核(`verify-game24-findings` workflow):
- C1(50% 天花板)与 C2(GRPO 坍缩)均判为 **SOUND**;验证器无假阳性(已补 `**`/`//`)、train/test 数字多重集 0 重叠、无 prompt 示例泄漏、无 max_new_tokens 截断伪影。
- caveat:50% 写作 **~50%±10pp**(单次 best-of-64 抽样);"可部署"需 64×+验证器;坍缩端点显著、逐步单调不下定论。eval 已加随机种子。

## 7.6 提升账本
| 指标 | 起点 | 现在最好 | 手段 |
|---|---|---|---|
| 难测试 可部署求解率 | 1%(贪心) | **~50%**(bv@64,±10pp) | SFT v2 + 高温 + best-of-N(GRPO 宜轻) |
| 分布内 greedy | 27% | **42%** | SFT v2 + sparse GRPO |
| 平衡(分布内+best-of-N) | — | 31% / bv@16 18% | SFT v2 + 多样性保护 GRPO@20 |

**结论**:GRPO 是 RLVR 的"利用"算子(把分布内做尖),而难题突破靠"探索"(强 SFT 的多样性 + 验证器 best-of-N)。最佳工程方案是两者结合,且 GRPO 需用高 KL/早停来避免牺牲 best-of-N。

---

# 8. 训练监控与曲线（对应评分点"监控 reward/正确率曲线，必要时调超参"）

## 8.1 监控方式
本机到 wandb.ai 的在线同步不稳定,故训练采用 **console logger**(`trainer.logger=['console']`)逐步打印指标,
并设 `WANDB_MODE=offline`(离线 run 存于 `wandb/`,可日后 `wandb sync` 上传)。
**所有训练曲线来自解析 console 日志**(`game24-grpo-*.log`),用 `scripts/analyze_grpo.py` 抽成
CSV + matplotlib 图,无需依赖 wandb 在线面板。

## 8.2 每步监控的指标(verl 逐步输出)
| 指标 | 含义 | 我们看它判断什么 |
|---|---|---|
| `critic/score/mean` | 每步平均奖励(核心) | 是否在学(应从 ~0.06 升;SFT 起点已 ~0.3) |
| `critic/score/max` | 批内最高奖励 | 是否出现 1.0(组内有正确样本→有梯度) |
| `val/test_score/game24` | 难测试集验证分(每 `test_freq=20` 步) | 泛化趋势(注:sparse/shaped 不可跨模式比) |
| `actor/kl_loss` | 对参考策略 KL | 是否漂移过大(用它做高 KL 锚定实验) |
| **`actor/entropy_loss`** | 策略熵 | **多样性坍缩的直接观测量**(见 8.3) |
| `response_length/mean` | 平均输出长度 | 是否截断/退化 |
| `actor/grad_norm` | 梯度范数 | 训练稳定性 |

## 8.3 三张报告图(`results/`)
- **`curves_phase1.png`**：纯 GRPO vs SFT→GRPO 的 `critic/score/mean` 与 `val/test_score` ——
  纯 GRPO 长期卡在 ~0.10–0.13,SFT→GRPO 从第 1 步就 0.25–0.44。直观展示"零梯度组"被 SFT 解决。
- **`curves_wave1.png`**：奖励设计消融(sparse-n4 / sparse-n8 / shaped / staged)的奖励曲线 ——
  shaped 因部分分天然更高(~0.5),sparse ~0.27;说明 shaped 给了更稠密的逐样本信号。
- **`curves_entropy.png`**：`actor/entropy_loss` 曲线(sparse vs divC 高KL vs divD 高熵)——
  **这是"GRPO 坍缩 best-of-N"的机制图**:sparse 的熵快速跌向 0(策略尖锐化),高 KL/高熵配置熵衰减更慢,
  对应它们 best-of-N 保留更好。把 §7.3 的多样性结论与这条熵曲线对应起来即可。

## 8.4 "必要时调超参"的体现
监控驱动了整个 sweep:观察到纯 GRPO 奖励停滞 → 上 SFT 暖启动;观察到 best-of-N 随训练坍缩 +
熵快速下降 → 设计高 KL/高熵/早停的多样性保护臂(§7.3);观察到 shaped 奖励曲线更稠密 → 纳入消融。
超参网格与逐项结果见 `EXPERIMENTS.md` §5。
