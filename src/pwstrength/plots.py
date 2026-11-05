"""Matplotlib figures for the results/ folder.

Style notes: one shared light theme, a fixed categorical order (class or
feature-group identity always gets the same colour), a single-hue sequential
ramp for the heatmaps, thin marks and a recessive grid.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .labeler import LABELS, STRONG_MIN_BITS, WEAK_MAX_BITS

# --- palette (fixed order; see README "Design choices") ---
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#9a9890"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]           # slot 1, 2, 3
CLASS_COLORS = dict(zip(LABELS, CAT))             # weak, medium, strong
GROUP_COLORS = {"composition": CAT[0], "entropy": CAT[1], "pattern": CAT[2]}
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                 "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": NEUTRAL, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2,
    "ytick.color": TEXT_2, "text.color": TEXT, "axes.titlecolor": TEXT,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "font.family": "DejaVu Sans", "figure.dpi": 100,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def entropy_distributions(df, path):
    """Per-class histograms of three entropy measures."""
    panels = [
        ("shannon_entropy", "Shannon entropy (bits / char)", None),
        ("pool_entropy_bits", "Pool entropy: length × log2(alphabet) (bits)", (WEAK_MAX_BITS, STRONG_MIN_BITS)),
        ("guess_bits", "Pattern-aware guess estimate (bits, used by labeler)", (WEAK_MAX_BITS, STRONG_MIN_BITS)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, (col, title, thr) in zip(axes, panels):
        hi = np.percentile(df[col], 99.5)
        bins = np.linspace(0, hi, 50)
        for k, name in enumerate(LABELS):
            vals = df.loc[df.label == k, col].clip(upper=hi)
            ax.hist(vals, bins=bins, histtype="step", linewidth=2, color=CLASS_COLORS[name],
                    label=f"{name} (n={len(vals):,})", density=True)
        if thr:
            for t in thr:
                ax.axvline(t, color=TEXT_2, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(thr[0], ax.get_ylim()[1] * 0.95, f" {thr[0]:.0f}", color=TEXT_2, fontsize=8, va="top")
            ax.text(thr[1], ax.get_ylim()[1] * 0.95, f" {thr[1]:.0f}", color=TEXT_2, fontsize=8, va="top")
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("density")
        ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper left")
    fig.suptitle("Entropy distributions by strength class", fontsize=12, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def confusion_heatmap(cm, title, path):
    """Single confusion-matrix heatmap with counts and row-normalised percentages."""
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    _draw_cm(ax, cm, title)
    _save(fig, path)


def _draw_cm(ax, cm, title):
    rown = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(rown, cmap=SEQ, vmin=0, vmax=1)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(3):
        for j in range(3):
            dark = rown[i, j] > 0.55
            ax.text(j, i, f"{cm[i, j]:,}\n{rown[i, j]:.1%}", ha="center", va="center",
                    fontsize=8.5, color="#ffffff" if dark else TEXT)
    ax.set_xticks(range(3), LABELS)
    ax.set_yticks(range(3), LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)


def confusion_grid(cms: dict, path):
    """Confusion matrices for several models side by side."""
    fig, axes = plt.subplots(1, len(cms), figsize=(3.8 * len(cms), 3.8))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        _draw_cm(ax, cm, name)
    _save(fig, path)


def feature_importance(imp_df, groups: dict, path, top: int = 18, title=""):
    """Horizontal bars of permutation importance (mean ± std), coloured by feature group."""
    d = imp_df.sort_values("importance_mean", ascending=False).head(top).iloc[::-1]
    group_of = {f: g for g, fs in groups.items() for f in fs}
    colors = [GROUP_COLORS[group_of[f]] for f in d.feature]
    fig, ax = plt.subplots(figsize=(7.2, 0.32 * len(d) + 1.2))
    ax.barh(d.feature, d.importance_mean, xerr=d.importance_std, color=colors, height=0.62,
            error_kw={"ecolor": TEXT_2, "elinewidth": 0.8, "capsize": 2})
    for y, (v, sd) in enumerate(zip(d.importance_mean, d.importance_std)):
        ax.text(v + sd + d.importance_mean.max() * 0.012, y, f"{v:.3f}", va="center", fontsize=7.5, color=TEXT_2)
    ax.set_xlabel("drop in macro-F1 when feature is permuted (test set)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in GROUP_COLORS.values()]
    ax.legend(handles, GROUP_COLORS.keys(), loc="lower right", title="feature group", title_fontsize=8.5)
    ax.set_title(title, loc="left")
    _save(fig, path)


def model_comparison(metrics_df, path):
    """Macro-F1 and accuracy for learned models vs baselines (two small panels, one axis each)."""
    d = metrics_df.sort_values("f1_macro")
    is_base = d.model.str.startswith("Baseline")
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(d) + 1.4), sharey=True)
    for ax, col, label in zip(axes, ["f1_macro", "accuracy"], ["macro-F1", "accuracy"]):
        ax.barh(d.model, d[col], color=np.where(is_base, NEUTRAL, CAT[0]), height=0.6)
        for y, v in enumerate(d[col]):
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=TEXT_2)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel(label + " (test set)")
        ax.grid(axis="y", visible=False)
    axes[0].set_title("Learned models (blue) vs rule baselines (gray)", loc="left")
    _save(fig, path)


def ablation(ab_df, path):
    """Macro-F1 of the random forest as feature groups are added."""
    d = ab_df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.5 * len(d) + 1.2))
    ax.barh(d.feature_set, d.f1_macro, xerr=d.cv_f1_std, color=CAT[0], height=0.6,
            error_kw={"ecolor": TEXT_2, "elinewidth": 0.8, "capsize": 2})
    for y, v in enumerate(d.f1_macro):
        ax.text(v + 0.015, y, f"{v:.3f}", va="center", fontsize=8, color=TEXT_2)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("macro-F1 (test set; whiskers = 5-fold CV std)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Feature-group ablation (RandomForest)", loc="left")
    _save(fig, path)


def pattern_prevalence(prev_df, path):
    """Share of passwords in each class that exhibit each weakness pattern."""
    feats = list(prev_df.index)
    y = np.arange(len(feats))
    h = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(feats) + 1.3))
    for k, name in enumerate(LABELS):
        ax.barh(y + (k - 1) * h, prev_df[name], height=h - 0.03, color=CLASS_COLORS[name], label=name)
    ax.set_yticks(y, feats)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("share of passwords in class")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", title="class", title_fontsize=8.5)
    ax.set_title("Weakness-pattern prevalence by strength class", loc="left")
    _save(fig, path)


def lofo(lofo_df, path):
    """Accuracy on each password family when that family is held out of training."""
    d = lofo_df.sort_values("accuracy_heldout")
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(d) + 1.3))
    ax.barh(d.family, d.accuracy_in_dist, color=NEUTRAL, height=0.36, label="family seen in training (5-fold CV)")
    ax.barh(np.arange(len(d)) - 0.38, d.accuracy_heldout, color=CAT[0], height=0.36,
            label="family held out of training")
    ax.set_yticks(np.arange(len(d)) - 0.19, d.family)
    for y, v in enumerate(d.accuracy_heldout):
        ax.text(v + 0.01, y - 0.38, f"{v:.3f}", va="center", fontsize=7.5, color=TEXT_2)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("accuracy on the family (HistGradientBoosting)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_title("Leave-one-family-out generalisation", loc="left")
    _save(fig, path)
