# 实验报告严格审阅（review_report.md）

> 审阅对象：`NLP_实验报告_选题三_基于强化学习的24点游戏求解.docx`
> 范围：**只列"逻辑不自洽"或"没说明白"的点**，并对每条给出**本项目代码/结果中的实际信息**作为修正依据。
> 依据文件：`examples/data_preprocess/game24.py`、`scripts/train_game24_grpo_hf.sh`、`verl/utils/reward_score/game24.py`、
> `results/comparison.json`、`results/ablation_summary.md`、`results/ceiling/*.log`、`results/divsweep/*`、`REPORT.md`、`V1_vs_V2.md`。

---

## 一、严重问题（事实/逻辑层面，建议必改）

### 1. 数据集来源表述与实现不一致（且自相矛盾）
- **报告原文**：摘要/1.2/3.2 称"训练使用 **nlile/24-game (solvable=True) 1262 条**"；幻觉集写"**nlile/24-game (solvable=False)** … 合成无解四元组 100"。
- **为何有问题**：
  1. **训练集来源错**：实际 `game24.py` 只 `load_dataset("test-time-compute/game-of-24")`，**train 和 test 同源**，按 `Rank` 切分（train=Rank 1–1262、test=Rank 1263–1362）。**全程没有加载 nlile/24-game**。
  2. **幻觉集自相矛盾**："nlile solvable=False"与"合成无解四元组"是两回事；实际是**后者**——从 `{1..13}⁴` 组合里减去可解集合成 100 个无解四元组（代码注释明确："nlile/24-game no longer contains solvable=False rows"），并已用 `solve24()` 验证 100 题**全部无解**。
  3. 3.2 表把**同一个来源**（test-time-compute）拆成"训练=nlile / 测试=ttc"两个来源。
- **本项目实际信息（修正依据）**：训练/测试均来自 `test-time-compute/game-of-24`，按 Rank 难度切分；幻觉集为合成无解四元组。
  > 缓和说明：两数据集题目内容相同（都是那 1362 道），所以**结果数值不受影响**，但**出处必须改正**为 test-time-compute + 合成幻觉，否则与代码不符、且自相矛盾。

### 2. 推理引擎与硬件描述与实际不符（"声称做了但没做"）
- **报告原文**：3.1 "**vLLM 作为推理引擎**"；"硬件环境：**单卡 NVIDIA RTX 3060 12GB**"。
- **为何有问题**：本环境 **vLLM ABI 损坏不可用**（`undefined symbol: cuTensorMapEncodeTiled`，cu118 与 cu12.x 不匹配），实际改用 **HF rollout**（`actor_rollout_ref.rollout.name=hf`），并为此给 `verl/workers/rollout/hf_rollout.py` 打了 **n>1 采样补丁**、加了 `VERL_DISABLE_FLASH_ATTN_CE` 开关。硬件是 **8×A100-80GB**（用隔离 Ray 集群并行跑多个 arm）。"RTX 3060/vLLM" 是 TinyZero 参考 README 的设定，不是本次实跑。
- **本项目实际信息**：rollout 后端 = HF rollout；GPU = 8×A100-80GB；这正是本项目相对参考实现的主要工程改动，**应写进去而不是抹掉**。

### 3. 超参表与实跑配置/正文步数不一致
- **报告原文**：3.3 表 `Train Batch Size=32`、`Max Resp Length=512`、`Total Epochs=15`；而 4.1 又说"训练 **160 步**"。
- **为何不自洽**：`32 batch × 15 epoch × 1262 题 ÷ 32 ≈ 590 步`，与"160 步"对不上。实跑（`train_game24_grpo_hf.sh`）是 **batch=16、max_response=256(v1)/320(v2)、`total_epochs=3` 跑到 160 步**（因步数判定 off-by-one，实际存到 `global_step_120`）。`32 / 512 / 15` 是参考脚本默认值，非本次实跑。
- **本项目实际信息**：把超参表改成实跑值（batch 16、resp 256/320、epochs 3≈160 步、lr 1e-6、KL 0.001、ent 0.001、T 1.0、rollout=hf）。

---

## 二、表述不清 / 易误读（建议澄清）

### 4. "SFT 使解题率提升到 20–40%" 未限定口径
- **报告原文**：2.3 "SFT 冷启动 … 使解题率提升到 20–40%"。
- **没说明白**：这是**分布内（训练分布）贪心准确率**（SFT-only v1=20%、SFT v2=27%、其上 GRPO control80=42%）。**难测试集贪心仍≈1%**（表1）。不限定口径会被读成"难题也 20–40%"。
- **修正依据**：明确写"分布内 20–42%；难测试集（最难 100）贪心≈1%"。

### 5. "多样性"结论与训练步数自相矛盾（结论口径过强）
- **报告原文**：5.1(3) 把"提高 KL（DivC）是解决熵坍缩的**有效手段**"列为确定结论；5.2(5) 却承认"DivA–DivD **仅训练 20–40 步**"。
- **为何不自洽**：`results/divsweep/*20.log` 显示 div 系列只有 step20（个别 step40），从 20–40 步下"有效手段"的结论偏强，应降级为"初步观察"。
- **更硬的证据（建议改用）**：`results/ablation_summary.md` 显示 **sftv2base bv@16=21% → control40=12% → control80=5%**，即**GRPO 训练步数越多、best-of-N 反而越差**（熵坍缩对 best-of-N 的破坏），这条是真实、可核验的主结论，应突出；而 DivC 的优势只到"初步"。

### 6. Countdown OOD 的样本数与"OOD 比域内高"未解释
- **报告原文**：3.2 标 Countdown 样本数 **1024**；表1 给 12.89% 等。
- **没说明白**：实际评测用 `eval_compare --countdown_n 256`，**只评前 256 条**，不是 1024。且 Countdown OOD（12.89%）**高于**难测试（1%），报告未解释——因为这 256 条 Countdown 的难度分布与"最难 100 题"不同，所以"OOD 比域内高"并不矛盾，但**必须说明**，否则读着像逻辑错误。
- **修正依据**：注明"Countdown 评测取 256 条子样本"，并加一句难度分布差异的说明。

### 7. 图表引用与实际产出对不上
- **报告原文**：引用 **图1–图6**。
- **没说明白**：仓库 `results/` 下只确认有 **3 张曲线图**：`curves_phase1.png`（图1）、`curves_wave1.png`（图2）、`curves_entropy.png`（图3）。**图4（best-of-N 天花板）、图5（综合对比）、图6（多样性 bv@k）未见对应 PNG**（其数据在 `ablation_summary.md`/`ceiling/*.json`）。需确认这三张图确已生成并嵌入 docx。
- 另：表2（Wave-1，bv@8≤4%）与 图6/4.6（DivC bv@8=12%）**不是同一组 arm、不同步数**，报告并列时未说明基准差异，易显矛盾。

### 8. 引用小错
- **报告原文**：3.1 "系统架构包含以下组件**[11]**"，而 [11] 是 **TRL**。
- **问题**：实际框架是 **veRL[10]**，本项目没用 TRL。架构组件引用应指 [10]。

---

## 三、已核实属实、**不是问题**（供放心，勿误删）
- **"基模 1%、n=4 每组至少一条正确 ≈4%"**：`1-(0.99)^4 ≈ 3.9%`，自洽。
- **"T=1.5、k=64 达 50%"有据**：`results/ceiling/T1.5.log` `best_verified@64=0.5`（且 T1.0=37%、T1.2=39%），非杜撰；可在报告标注数据出处 `results/ceiling/`。
- **表1 主结论数字属实**：`results/comparison.json` 与表1一致（SFT→GRPO 分布内 27%、Countdown OOD 12.89%、纯 GRPO 2.34%、幻觉全 0%）。SFT→GRPO 相对纯 GRPO 的优势成立。

---

## 四、一句话总结
报告的**核心结论与主要数字是站得住的**（SFT 冷启动显著优于纯 GRPO、best-of-N 天花板 50%、熵坍缩现象），但**实现描述层**有三处必须改的硬伤——**数据来源（nlile→实为 test-time-compute + 合成幻觉）、推理引擎/硬件（vLLM/3060→实为 HF rollout/A100）、超参表（32/512/15→实为 16/256/3epoch≈160步）**；外加若干口径未限定与图表对齐问题。改完这些，报告即与代码自洽。
