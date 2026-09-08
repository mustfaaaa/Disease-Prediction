"""Exploratory data analysis (Phase 5).

Every figure here answers a question that later shapes a modelling decision -
nothing is drawn for decoration.  Figures are written to ``outputs/figures``
and the numeric findings are returned so the report can quote them.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C

PALETTE = {
    "neg": "#5B8DB8",     # negative class
    "pos": "#C9772E",     # positive class
    "grid": "#D9DEE5",
    "ink": "#26303B",
    "muted": "#6B7885",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelcolor": PALETTE["ink"],
    "axes.edgecolor": PALETTE["grid"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": PALETTE["ink"],
    "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"],
    "grid.color": PALETTE["grid"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def _save(fig, name: str) -> str:
    path = C.FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path.relative_to(C.ROOT)).replace("\\", "/")


def _cat_labels(feature: str) -> dict:
    f = C.FEATURE_BY_NAME[feature]
    return {float(c["value"]): c["label"] for c in f.get("choices", [])}


# --------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> dict:
    """Produce all EDA figures and return the findings behind them."""
    findings: dict = {"figures": {}, "insights": {}}
    y = df[C.TARGET]

    # 1 -------------------------------------------------- target distribution
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    bars = ax.bar([C.TARGET_LABELS[int(k)] for k in counts.index], counts.values,
                  color=[PALETTE["neg"], PALETTE["pos"]], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v}\n({v/len(y)*100:.1f}%)",
                ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("Patients")
    ax.set_title("Target distribution (n = %d)" % len(y))
    ax.yaxis.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    findings["figures"]["target_distribution"] = _save(fig, "01_target_distribution.png")
    findings["insights"]["class_balance"] = {
        "counts": {int(k): int(v) for k, v in counts.items()},
        "minority_share": round(float(counts.min() / counts.sum()), 4),
        "imbalance_ratio": round(float(counts.max() / counts.min()), 3),
        "reading": ("Near balanced (1.18:1). Heavy re-sampling such as SMOTE is "
                    "not indicated; stratified splits plus optional class "
                    "weights are enough."),
    }

    # 2 ------------------------------------------------ numeric distributions
    nums = C.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, col in zip(axes.ravel(), nums):
        for cls, colour in ((0, PALETTE["neg"]), (1, PALETTE["pos"])):
            ax.hist(df.loc[y == cls, col].dropna(), bins=18, alpha=0.62,
                    color=colour, label=C.TARGET_LABELS[cls], edgecolor="white",
                    linewidth=0.4)
        ax.set_title(C.FEATURE_BY_NAME[col]["label"], fontsize=9)
        unit = C.FEATURE_BY_NAME[col]["unit"]
        ax.set_xlabel(unit or "")
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
    axes.ravel()[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Numeric feature distributions by outcome", y=1.01, fontsize=11)
    findings["figures"]["numeric_distributions"] = _save(fig, "02_numeric_distributions.png")

    # 3 ------------------------------------------------------ box plots
    fig, axes = plt.subplots(1, len(nums), figsize=(13, 3.4))
    sep = {}
    for ax, col in zip(axes, nums):
        data = [df.loc[y == 0, col].dropna(), df.loc[y == 1, col].dropna()]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        tick_labels=["Neg", "Pos"], medianprops=dict(color=PALETTE["ink"]))
        for patch, colour in zip(bp["boxes"], [PALETTE["neg"], PALETTE["pos"]]):
            patch.set_facecolor(colour)
            patch.set_alpha(0.65)
        ax.set_title(C.FEATURE_BY_NAME[col]["label"], fontsize=8.5)
        ax.yaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        m0, m1 = data[0].mean(), data[1].mean()
        pooled = np.sqrt((data[0].var(ddof=1) + data[1].var(ddof=1)) / 2)
        sep[col] = {
            "mean_negative": round(float(m0), 2),
            "mean_positive": round(float(m1), 2),
            "cohens_d": round(float((m1 - m0) / pooled), 3) if pooled else 0.0,
        }
    fig.suptitle("Numeric features by outcome (box plots)", y=1.04, fontsize=11)
    findings["figures"]["boxplots"] = _save(fig, "03_boxplots_by_target.png")
    findings["insights"]["numeric_separation"] = sep

    # 4 --------------------------------------------------- correlation heatmap
    corr = df[C.FEATURE_ORDER + [C.TARGET]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [C.FEATURE_BY_NAME[c]["label"] if c in C.FEATURE_BY_NAME else "Outcome"
              for c in corr.columns]
    short = [l if len(l) < 22 else l[:20] + "..." for l in labels]
    ax.set_xticks(range(len(short)), short, rotation=55, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(short)), short, fontsize=7.5)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            if abs(v) >= 0.30:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(v) > 0.6 else PALETTE["ink"])
    fig.colorbar(im, ax=ax, shrink=0.75, label="Pearson r")
    ax.set_title("Feature correlation matrix")
    findings["figures"]["correlation"] = _save(fig, "04_correlation_heatmap.png")

    tcorr = corr[C.TARGET].drop(C.TARGET).sort_values(key=abs, ascending=False)
    findings["insights"]["target_correlation"] = {
        k: round(float(v), 3) for k, v in tcorr.items()
    }
    off = corr.drop(index=C.TARGET, columns=C.TARGET)
    names = list(off.columns)
    mat = np.abs(np.array(off.to_numpy(), dtype=float, copy=True))
    iu = np.triu_indices_from(mat, k=1)
    order = np.argsort(mat[iu])[::-1][:3]
    findings["insights"]["strongest_feature_pairs"] = {
        f"{names[iu[0][o]]} ~ {names[iu[1][o]]}": round(float(mat[iu][o]), 3)
        for o in order
    }

    # 5 ------------------------------------ categorical outcome-rate analysis
    cats = C.NOMINAL_FEATURES + C.BINARY_FEATURES
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    rates = {}
    for ax, col in zip(axes.ravel(), cats):
        grp = df.groupby(col, observed=True)[C.TARGET].agg(["mean", "size"])
        lut = _cat_labels(col)
        names = [lut.get(float(i), str(i)) for i in grp.index]
        names = [n if len(n) < 16 else n[:14] + "..." for n in names]
        colours = [PALETTE["pos"] if m >= 0.5 else PALETTE["neg"] for m in grp["mean"]]
        ax.bar(names, grp["mean"] * 100, color=colours, width=0.6)
        for i, (m, s) in enumerate(zip(grp["mean"], grp["size"])):
            ax.text(i, m * 100 + 2, f"{m*100:.0f}%\nn={s}", ha="center",
                    va="bottom", fontsize=7)
        ax.axhline(y.mean() * 100, ls="--", lw=1, color=PALETTE["muted"])
        ax.set_ylim(0, 118)
        ax.set_title(C.FEATURE_BY_NAME[col]["label"], fontsize=8.5)
        ax.tick_params(axis="x", labelrotation=25, labelsize=7)
        ax.set_ylabel("% positive" if ax in axes[:, 0] else "")
        rates[col] = {str(lut.get(float(i), i)): round(float(m), 3)
                      for i, m in grp["mean"].items()}
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    fig.suptitle("Positive-outcome rate by category (dashed line = cohort base rate "
                 f"{y.mean()*100:.1f}%)", y=1.02, fontsize=11)
    findings["figures"]["categorical_rates"] = _save(fig, "05_categorical_outcome_rates.png")
    findings["insights"]["categorical_positive_rates"] = rates

    # 6 --------------------------------------------------------- missingness
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        ax.barh([C.FEATURE_BY_NAME[c]["label"] for c in miss.index], miss.values,
                color=PALETTE["neg"], height=0.5)
        for i, v in enumerate(miss.values):
            ax.text(v + 0.05, i, f"{v} ({v/len(df)*100:.1f}%)", va="center", fontsize=8)
        ax.set_xlim(0, miss.max() * 1.7)
        ax.set_xlabel("Missing values")
        ax.set_title("Missing data by feature")
        ax.xaxis.grid(True, alpha=0.4)
        ax.set_axisbelow(True)
        findings["figures"]["missingness"] = _save(fig, "06_missingness.png")
    findings["insights"]["missing_per_column"] = {k: int(v) for k, v in miss.items()}

    (C.METRICS / "eda_findings.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8")
    return findings
