# 24 点 SFT→GRPO 实验：V1 vs V2 版本对照

> 框架：TinyZero（= veRL）。Backbone：Qwen2.5-1.5B-Instruct。数据：`test-time-compute/game-of-24`
> 同源按难度切分（train=最易 1262 / test=最难 100 / 幻觉=100 合成无解），加 Countdown-3to4 作 OOD。
> 本文说明两轮实验（V1 / V2）的**全部差异**与**当前状态**。

## 0. 一句话区分

- **V1**：第一轮**已完成、已评测、已写进 `REPORT.md`** 的受控对比（纯 GRPO vs SFT→GRPO）。报告里所有数字都是 V1。
- **V2**：第二轮"刷分"——**更强的 SFT v2 + 超参/奖励 sweep + best-of-N 评测**。目前**只跑了一部分、评测未收尾、未进报告**。

---

## 1. 差异速览

| 维度 | **V1** | **V2** |
|---|---|---|
| SFT 数据 | `data/sft_game24`，1199 条，**每题 1 解**（最短） | `data/sft_game24_v2`，**2391 条（≈2×）**，**每题多解**（最短+中等复杂度） |
| SFT 失败示例 | 2 个（基础加减乘） | **3 个**，含 1 个**分数/非整数**尝试（如 `12/(10/5)=6`） |
| SFT 自检行 | 无 | **有**：`Check: <expr> = 24. Uses [...] once each. Correct.` |
| SFT 超参 | lr=2e-5, epochs=3 | **lr=1e-5, epochs=4** |
| SFT ckpt | `game24-sft/global_step_111` | `game24-sft-v2/global_step_{74,148,222,296}` |
| 奖励函数 | 固定稀疏 `1.0 / 0.1 / 0.0` | **env 门控**：`GAME24_REWARD=sparse`(默认)或 `shaped`(近似分)；`solved_rate` 入 `ground_truth` |
| GRPO max_response | 256 | **320** |
| GRPO 起点 | base（纯GRPO arm）或 SFT v1（SFT→GRPO arm） | 一律 **SFT v2 (step_296)** |
| GRPO 变体 | 2 个：纯GRPO / SFT→GRPO（只差初始权重） | **sweep**：control / n8 / shaped / staged-p1 / divA–D（扫 n 与 KL） |
| 评测 | 贪心 + pass@4 | **新增 best_verified@k（验证器引导 best-of-N，可部署）+ maj@k** |
| 状态 | ✅ 完成 + 评测 + 报告 | ⏳ 部分训练（step 20–89，未到 160）+ 评测未收尾 |

---

## 2. SFT 细节

**V1**（`examples/data_preprocess/generate_sft_game24.py` 原版）：每题用精确求解器取**最短解**，固定 2 个失败尝试 + 答案；1199 条；lr 2e-5 / 3 epochs。

**V2**（同文件已改）：每题输出**≥2 条**（最短 + 一条中等复杂度，偏分数/除法路径），失败尝试增到 3 个（含 1 个非整数中间值），并在解后加一行**数字用法自检**。共 2391 条（2270 行与他行共享同题 prompt → 同题多解）；lr 1e-5 / 4 epochs。一条 V2 样本：

```
Let me solve this step by step.
<think>
I need to make 24 from [4, 5, 10, 12], using each number exactly once.
Let me try some combinations first:
- Try 12 + 10 + 5 + 4 = 31 ≠ 24. Does not work.
- Try 12 * 10 - 5 - 4 = 111 ≠ 24. Does not work.
- Try 12 / (10 / 5) = 6 ≠ 24. Does not work.        ← 分数/非整数失败示例(V2 新增)
Let me think differently:
4 * 5 / 10 * 12 = 24 ✓  Found it!
Check: 4 * 5 / 10 * 12 = 24. Uses [4, 5, 10, 12] once each. Correct.   ← 自检行(V2 新增)
</think>
<answer>4 * 5 / 10 * 12</answer>
```

**直接影响**：SFT v2 是更强的起点——GRPO 第 1 步 `critic/score/mean≈0.48`（V1 同点 ~0.27），响应更长（~191 vs ~103 token）。

---

## 3. 奖励函数（`verl/utils/reward_score/game24.py`）

- **V1**：`正确=1.0 / 有<answer>但错=0.1 / 无答案=0.0`，固定。
- **V2**：按环境变量 `GAME24_REWARD` 切换（默认 `sparse`，等价 V1）：
  - `shaped`：数字用对但值≠24 时给 `0.1 + 0.5·max(0,1-|v-24|/24)`（封顶 0.6，满分仍 1.0），仅在 `validate_numbers` 通过时生效（无解题/错数字拿不到近似分 → 防刷分）。
  - 另把 `solved_rate` 写进 `ground_truth`，为难度加权预留。
  - **评测默认不设该变量（走 sparse）**，所以准确率/ best_verified 口径与 V1 一致、不被塑形抬高。

---

## 4. GRPO sweep（V2 新增；均从 SFT v2 起跑，2 GPU/arm 隔离 Ray 并行）

| arm | rollout.n | KL coef | reward | 跑到 step | ckpts |
|---|---|---|---|---|---|
| control | 4 | 0.001 | sparse | 89 | 40,80 |
| n8 | 8 | 0.001 | sparse | 82 | 40,80 |
| shaped | 4 | 0.001 | shaped(近似分) | 88 | 40,80 |
| staged-p1 | 4 | 0.001 | shaped(分段第一段) | 89 | 40,80 |
| divA | 4 | **0.005** | — | 40 | 20,40 |
| divB | 8 | **0.010** | — | 32 | 20 |
| divC | 4 | **0.030** | — | 32 | 20 |
| divD | 4 | **0.005** | — | 32 | 20 |

> 即 V2 扫了 **rollout.n(4/8)**、**KL 系数(0.001/0.005/0.01/0.03)**、**奖励设计(sparse/shaped/staged)**。
> **未扫 learning rate**——所有 GRPO 运行(V1+V2)actor lr 都固定 1e-6。

---

## 5. 结果现状

### V1（完整，`results/comparison.json` + `REPORT.md`；checkpoint=step_120，贪心口径）
| Model | in-dist(train) | hard-test greedy | pass@4 | mean | 幻觉↓ | Countdown OOD |
|---|---|---|---|---|---|---|
| base | 0% | 0% | 0% | 0.091 | 0% | 1.95% |
| 纯 GRPO | 5% | 1% | 0% | 0.109 | 0% | 2.34% |
| SFT-only | 20% | 1% | 3% | 0.109 | 0% | 5.08% |
| **SFT→GRPO** | **27%** | 1% | **5%** | 0.109 | 0% | **12.89%** |

结论（V1）：SFT→GRPO 在分布内(5.4×)与 OOD(5.5×)上显著优于纯 GRPO；难测试贪心两者都 ~0-1%(1.5B 1-shot 天花板)。

### V2（**部分**，`results/comparison.md` 当前只评了 1 个 checkpoint）
唯一已评测：`divB @ step20`（SFT v2 + n=8 + KL=0.01），用 best-of-N-verified：

| 模型 | in-dist | hard-test greedy | bv@1 | bv@4 | bv@8 | **bv@16** | maj@16 |
|---|---|---|---|---|---|---|---|
| divB20 | **32%** | 0% | 2% | 6% | 13% | **16%** | 1% |

- 信号正面：in-dist 27%→**32%**；难测试可部署求解率(best-of-N-verified)从 V1 的 `pass@4=5%` 提到 **bv@16=16%**。
- 但这是**单个、未跑满(step20)的 checkpoint**，幻觉/Countdown 该行还是 nan，**不能当最终结论**。

> 注：`best_verified@k` = 采样 k 个、用精确验证器输出其中正确的一个（可部署）；`maj@k` = 多数投票表达式是否正确。

---

## 6. 关键差异背后的"为什么"
- 难测试贪心接近 1.5B 单次上限 → V2 转而靠 **验证器引导 best-of-N** 把"可部署求解率"做大（最干净的提分点）。
- 纯 GRPO 卡在 0.10-0.13 是"零梯度组"问题 → V2 用**更强 SFT + 更多组采样(n8) + 近似分奖励**去密化信号。
- 数据红线：**不能扩训练难度**(全 1362 题里更难的就是测试集，扩=泄漏)；V2 因此只改轨迹质量，不动划分。

---

## 7. 未完成项（要让 V2 成结论需补）
1. **跑满 V2 sweep 到 160 步**（现 control/n8/shaped/staged ~88；divA–D 仅 20–40）。
2. **统一评测所有 V2 arm**（greedy + best_verified@8/16 + maj + 幻觉重验 0% + Countdown），出完整 V2 对照表。
3. **learning rate 消融**（目前完全没做）：固定其余、仅扫 `lr∈{5e-7,1e-6,2e-6,5e-6}`。
4. 选赢家，更新 `REPORT.md`（V1→V2 提升 + 各消融）。

## 8. 文件索引
- 报告：`REPORT.md`（V1）。
- V1 结果：`results/comparison.json`。曲线：`results/grpo_curves.{png,csv}`。定性：`results/qualitative_samples.md`。
- V2 部分评测：`results/comparison.md`（当前 divB20 一行）。
- SFT 数据：`data/sft_game24`(V1) / `data/sft_game24_v2`(V2)。
- 脚本：`scripts/{generate_sft_data_game24,train_sft_game24,train_game24_grpo_hf,eval_compare,analyze_grpo}.sh/.py`。
- checkpoints：`checkpoints/TinyZero/{game24-sft, game24-sft-v2, game24-grpo-base, game24-grpo-sftinit, game24-grpo-v2-*}`。
