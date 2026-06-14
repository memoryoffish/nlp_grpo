"""
Effect figures for the 24-game GRPO + self-critic study (compliant ToT test, ttc 900-999).

Produces results/figs/:
  fig_critic_prompt.png  - 4-way self-critic PROMPT ablation (logit / CoT / ToT-fewshot / verify-by-compute)
  fig_value_ladder.png   - greedy -> maj@16 -> selfcritic_cmp@16 -> best_verified@16 (generate >> verify)
  fig_verifier_flip.png  - verifier precision: naive (yes-man) vs verify-by-compute
  fig_ablation.png       - 5-axis GRPO ablation (greedy + best_verified@16)
  fig_overview.png       - all four panels in one 2x2 figure
"""
import json, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Chinese font (WenQuanYi available on this box)
for fn in ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei"]:
    if any(f.name == fn for f in fm.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [fn]; break
plt.rcParams["axes.unicode_minus"] = False

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIG = os.path.join(R, "figs"); os.makedirs(FIG, exist_ok=True)


def jload(name):
    p = os.path.join(R, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def load_prefixed(prefix, models):
    d = {}
    for m in models:
        p = os.path.join(R, f"{prefix}{m}.json")
        if os.path.exists(p):
            d.update(json.load(open(p)))
    return d


ALL = ["base", "SFT-single", "SFT-multi", "R1", "C", "A1", "A2", "A3", "A4a", "A4b", "A4c"]
KEY = ["SFT-multi", "C", "A1", "A4a", "A3"]

logit = jload("tot_selfcritic_compliant.json")          # selfcritic@16 (4 models)
cot = load_prefixed("cot_", KEY)                          # selfcritic_cot@16
tot = load_prefixed("tot2_", KEY)                         # selfcritic_tot@16
cmp = load_prefixed("cmp_", ALL)                          # verify-by-compute (up to 11 models)

C_LOGIT, C_COT, C_TOT, C_CMP = "#9aa7b1", "#6fa8dc", "#f6b26b", "#cc4125"
C_GREEDY, C_MAJ, C_SC, C_BV = "#b7b7b7", "#93c47d", "#cc4125", "#3d85c6"


def pct(ax):
    ax.set_ylim(0, 0.62); ax.yaxis.set_major_formatter(lambda x, _: f"{x*100:.0f}%")
    ax.grid(axis="y", ls=":", alpha=.5)


def panel_critic_prompt(ax):
    models = [m for m in KEY if m in cmp]
    x = np.arange(len(models)); w = 0.2
    g_logit = [logit.get(m, {}).get("selfcritic@16", np.nan) for m in models]
    g_cot = [cot.get(m, {}).get("selfcritic_cot@16", np.nan) for m in models]
    g_tot = [tot.get(m, {}).get("selfcritic_tot@16", np.nan) for m in models]
    g_cmp = [cmp.get(m, {}).get("selfcritic_cmp@16", np.nan) for m in models]
    ax.bar(x - 1.5 * w, g_logit, w, label="logit一词", color=C_LOGIT)
    ax.bar(x - 0.5 * w, g_cot, w, label="零样本 CoT", color=C_COT)
    ax.bar(x + 0.5 * w, g_tot, w, label="ToT few-shot", color=C_TOT)
    b = ax.bar(x + 1.5 * w, g_cmp, w, label="强制算 (本文)", color=C_CMP)
    ax.bar_label(b, fmt=lambda v: f"{v*100:.0f}%" if v == v else "", padding=2, fontsize=8)
    gr = [cmp.get(m, {}).get("greedy", np.nan) for m in models]
    ax.plot(x, gr, "k--", marker="o", ms=4, lw=1, label="greedy 基线")
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=0)
    ax.set_title("① 自评 critic 的 prompt 设计 → selfcritic@16(模型自评挑,无 oracle)")
    ax.legend(fontsize=8, ncol=2); pct(ax)


def panel_value_ladder(ax):
    models = [m for m in ALL if m in cmp]
    x = np.arange(len(models)); w = 0.2
    gr = [cmp[m].get("greedy", np.nan) for m in models]
    mj = [cmp[m].get("maj@16", np.nan) for m in models]
    sc = [cmp[m].get("selfcritic_cmp@16", np.nan) for m in models]
    bv = [cmp[m].get("best_verified@16", np.nan) for m in models]
    ax.bar(x - 1.5 * w, gr, w, label="greedy (1-shot)", color=C_GREEDY)
    ax.bar(x - 0.5 * w, mj, w, label="maj@16 多数投票", color=C_MAJ)
    ax.bar(x + 0.5 * w, sc, w, label="selfcritic@16 强制算(可部署)", color=C_SC)
    ax.bar(x + 1.5 * w, bv, w, label="best_verified@16 验证器上界", color=C_BV)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("② 价值阶梯:生成能力(bv上界)≫ 自评选择 ≫ 单发/投票")
    ax.legend(fontsize=8, ncol=2); pct(ax)


def panel_verifier_flip(ax):
    models = [m for m in KEY if m in cmp and m in cot]
    x = np.arange(len(models)); w = 0.27
    p_naive = [cot.get(m, {}).get("verifier_precision", np.nan) for m in models]
    p_tot = [tot.get(m, {}).get("verifier_precision", np.nan) for m in models]
    p_cmp = [cmp.get(m, {}).get("verifier_precision", np.nan) for m in models]
    ax.bar(x - w, p_naive, w, label="零样本CoT(老好人)", color=C_COT)
    ax.bar(x, p_tot, w, label="ToT few-shot", color=C_TOT)
    b = ax.bar(x + w, p_cmp, w, label="强制算", color=C_CMP)
    ax.bar_label(b, fmt=lambda v: f"{v*100:.0f}%" if v == v else "", padding=2, fontsize=8)
    ax.axhline(0.12, color="gray", ls=":", lw=1)
    ax.text(len(models) - 1, 0.13, "≈题目正确率(瞎说YES的精确率)", fontsize=7, ha="right", color="gray")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_title("③ 验证器精确率:从“老好人”(~12%)→ 真会判(45-66%)")
    ax.set_ylim(0, 0.8); ax.yaxis.set_major_formatter(lambda v, _: f"{v*100:.0f}%")
    ax.grid(axis="y", ls=":", alpha=.5); ax.legend(fontsize=8)


def panel_ablation(ax):
    # main compare eval (greedy + bv@16) for the ablation arms
    comp = jload("cmp_compliant_all.json")
    arm_lbl = {"base": "base", "SFT-multi": "SFT", "R1-puregrpo": "纯GRPO",
               "C-multi-n4-lr1e6": "C对照", "A1-n8": "A1:n8", "A2-single": "A2:单解",
               "A3-shaped": "A3:shaped", "A4a-lr5e7": "A4a:5e-7", "A4b-lr2e6": "A4b:2e-6",
               "A4c-lr5e6": "A4c:5e-6"}
    arms = [a for a in arm_lbl if a in comp]
    x = np.arange(len(arms)); w = 0.4
    gr = [comp[a]["test"]["greedy_acc"] for a in arms]
    bv = [comp[a]["test"].get("best_verified@16", np.nan) for a in arms]
    ax.bar(x - w / 2, gr, w, label="greedy", color=C_GREEDY)
    ax.bar(x + w / 2, bv, w, label="best_verified@16", color=C_BV)
    ax.set_xticks(x); ax.set_xticklabels([arm_lbl[a] for a in arms], rotation=30, ha="right")
    ax.set_title("④ 5轴 GRPO 消融(合规ToT测试集): SFT有无 / rollout_n / 多解 / reward / lr")
    ax.legend(fontsize=8); pct(ax)


def save_one(fn, drawfn, figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize)
    drawfn(ax); fig.tight_layout(); fig.savefig(os.path.join(FIG, fn), dpi=140); plt.close(fig)
    print("saved", fn)


def main():
    save_one("fig_critic_prompt.png", panel_critic_prompt)
    save_one("fig_value_ladder.png", panel_value_ladder, (11, 5))
    save_one("fig_verifier_flip.png", panel_verifier_flip)
    save_one("fig_ablation.png", panel_ablation, (11, 5))
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    panel_critic_prompt(axes[0, 0]); panel_value_ladder(axes[0, 1])
    panel_verifier_flip(axes[1, 0]); panel_ablation(axes[1, 1])
    fig.suptitle("24点:GRPO 消融 + 模型自评(ToT)效果总览 — 合规测试集 ttc 900-999, k=16",
                 fontsize=15, y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_overview.png"), dpi=140); plt.close(fig)
    print("saved fig_overview.png ->", FIG)


if __name__ == "__main__":
    main()
