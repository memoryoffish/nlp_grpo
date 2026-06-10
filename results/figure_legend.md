# 图 ↔ 配置 ↔ wandb 对照表（每条曲线是什么配置跑出来的）

> wandb entity/project：`3125567871-fisher-scientific / TinyZero`（10 个最新 run 带 `replay` tag）。
> 静态图在 `results/curves_*.png`，图例已直接写明每条线的配置；本表给出完整一一对应。

## 图 1 `results/curves_phase1.png` —— Phase-1 受控对比（仅初始权重不同）
四面板：critic/score/mean · val/test_score/game24 · response_length/mean · actor/kl_loss
| 曲线（图例） | 初始化 | 奖励 | n | lr | KL | 熵 | 温度 | 步数 | log / wandb |
|---|---|---|---|---|---|---|---|---|---|
| pure-GRPO | base（未训练） | sparse | 4 | 1e-6 | .001 | .001 | 1.0 | 160 | game24-grpo-base.log / run 4htqmxtc |
| SFT→GRPO | SFT v1 | sparse | 4 | 1e-6 | .001 | .001 | 1.0 | 160 | game24-grpo-sftinit.log / run hezcqpw2 |
**读图**：pure-GRPO 的 critic/score/mean 长期 0.10–0.13；SFT→GRPO 第 1 步即 0.25–0.44 → "零梯度组"被 SFT 解决。

## 图 2 `results/curves_wave1.png` —— Wave-1 奖励×采样数消融（均从 SFT v2）
| 曲线（图例） | 奖励 | n | 其余（同） | log / wandb |
|---|---|---|---|---|
| control | sparse | 4 | lr1e-6, KL.001, ent.001, T1.0 | game24-grpo-v2-control.log / run 97ezue98 |
| n8 | sparse | 8 | 同上 | game24-grpo-v2-n8.log / run 4vimu9wj |
| shaped | shaped | 4 | 同上 | game24-grpo-v2-shaped.log / run g83ai4rh |
| staged-p1 | shaped（→sparse 二段） | 4 | 同上 | game24-grpo-v2-staged-p1.log / run v6llz2wm |
**读图**：shaped 的 critic/score/mean 天然更高（~0.5，含部分分）；sparse ~0.27。注意 val 跨奖励模式不可比。

## 图 3 `results/curves_entropy.png` —— 多样性坍缩机制（策略熵 vs step，均从 SFT v2）
四面板：actor/entropy_loss · critic/score/mean · response_length/mean · actor/kl_loss
| 曲线（图例） | 奖励 | n | KL | 熵系数 | 温度 | log / wandb |
|---|---|---|---|---|---|---|
| control | sparse | 4 | .001 | .001 | 1.0 | game24-grpo-v2-control.log / run 97ezue98 |
| divC（高 KL 锚定） | shaped | 4 | **.03** | .01 | 1.2 | game24-grpo-v2-divC.log / run p3gy8v7j |
| divD（高熵探索） | shaped | 4 | .005 | **.03** | **1.3** | game24-grpo-v2-divD.log / run clbkq60f |
**读图**：control 的熵快速跌向 0（策略尖锐化 → best-of-N 坍缩）；divC/divD 熵衰减更慢（多样性保留更好）。

## 评测结果图（非训练曲线，离线评测）
- `results/ablation_summary.md`：A 表=SFT v2 best-of-N 天花板（温度×k）；B 表=探索/利用消融（含 div 各臂 step_20）。
- `results/comparison.{md,json}`：Phase-1 四模型综合对比。
- 评测原始 JSON：`results/{screen,ceiling,bv16,divsweep}/<name>.json`（文件名即模型/配置，如 `ceiling/T1.5.json`=温度1.5、`divsweep/divC20.json`=divC 第20步）。

## 其余两个 wandb run（也已回灌，未单独画静态图）
| run | 配置 | wandb |
|---|---|---|
| game24-grpo-v2-divA | shaped n4, KL.005, ent.01, T1.2 | run r20clu5r |
| game24-grpo-v2-divB | shaped n8, KL.01, ent.02, T1.2 | run vnsl2m1g |

> 完整超参网格见 `EXPERIMENTS.md` §5；报告引用见 `报告.md` §8（训练监控与曲线）。
