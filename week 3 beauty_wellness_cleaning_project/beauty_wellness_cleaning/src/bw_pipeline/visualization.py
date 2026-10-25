"""Charts for the report: pipeline flowchart, before/after comparisons, quality scorecard."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from . import config as C

PALETTE = {"before": "#D9534F", "after": "#2E8B8B", "ink": "#2B2B3A", "accent": "#B5517A", "soft": "#F6E8EE"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def _save(fig, name):
    C.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = C.FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_flowchart() -> "Path":
    stages = [
        ("1  INGEST", "Read raw CSV as text\n(no silent type guessing)", "#F6E8EE"),
        ("2  STANDARDISE", "snake_case columns\nplaceholder tokens -> NaN\nlabels -> canonical", "#F6E8EE"),
        ("3  DEDUPLICATE", "exact + near duplicates\n(customer, product, date)", "#E4F1F1"),
        ("4  PARSE & CONVERT", "price -> INR, size -> ml/g\nratings -> 1-5, dates\ntext, ingredients, PII hash", "#E4F1F1"),
        ("5  VALIDATE", "domain rules: age, price,\nsize, future dates\n-> invalid becomes NaN", "#FFF3D6"),
        ("6  RECONCILE", "product / customer master\nfill + repair conflicts\nprice vs product median", "#FFF3D6"),
        ("7  QUARANTINE", "no rating / bad date /\nreview before launch\n(saved for audit)", "#FBE0DF"),
        ("8  IMPUTE", "group median, 'Unknown',\nmissing-indicator flags", "#FBE0DF"),
        ("9  FEATURES", "price per 100ml, sentiment,\nseason, tiers, ingredient flags", "#E8E4F4"),
        ("10 OUTLIERS", "IQR + MAD + Isolation Forest\nflag, winsorise, keep original", "#E8E4F4"),
        ("11 SCALE & ENCODE", "log, Standard / Robust scaler\none-hot, train/test split", "#E4F1F1"),
        ("12 VALIDATE & EXPORT", "22 quality checks, manifest,\nclean + ML-ready files", "#F6E8EE"),
    ]
    fig, ax = plt.subplots(figsize=(13, 8.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.2); ax.axis("off")
    ax.text(6.5, 7.9, "Beauty & Wellness Data Cleaning Pipeline - Overall Flow", ha="center", fontsize=15, fontweight="bold", color=PALETTE["ink"])
    cols, w, h = 4, 2.75, 1.45
    xs = [0.45 + i * 3.15 for i in range(cols)]
    ys = [5.8, 3.5, 1.2]
    pos = []
    for i, (title, body, color) in enumerate(stages):
        r, c = divmod(i, cols)
        if r % 2 == 1:
            c = cols - 1 - c                       # snake layout
        x, y = xs[c], ys[r]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.12", fc=color, ec=PALETTE["ink"], lw=1.2))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="center", fontsize=10.5, fontweight="bold", color=PALETTE["accent"])
        ax.text(x + w / 2, y + 0.55, body, ha="center", va="center", fontsize=8.6, color=PALETTE["ink"])
        pos.append((x, y))
    for i in range(len(pos) - 1):
        (x1, y1), (x2, y2) = pos[i], pos[i + 1]
        if y1 == y2:
            a, b = ((x1 + w, y1 + h / 2), (x2, y2 + h / 2)) if x2 > x1 else ((x1, y1 + h / 2), (x2 + w, y2 + h / 2))
        else:
            a, b = (x1 + w / 2, y1), (x2 + w / 2, y2 + h)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=PALETTE["ink"]))
    ax.text(6.5, 0.5, "Every step logs rows in/out, missing cells and run-time  ->  cleaning_log.csv   |   Rejected rows  ->  quarantine_rows.csv   |   "
            "Seed + file hashes  ->  run_manifest.json", ha="center", fontsize=8.5, style="italic", color="#555")
    return _save(fig, "01_pipeline_flowchart.png")


def plot_missing(before: pd.Series, after: pd.Series):
    cols = [c for c in before.index if c in after.index and (before[c] > 0 or after[c] > 0)]
    d = pd.DataFrame({"before": before[cols], "after": after[cols]}).sort_values("before")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y = np.arange(len(d))
    ax.barh(y + 0.2, d["before"], 0.38, color=PALETTE["before"], label="Raw (after token normalisation)")
    ax.barh(y - 0.2, d["after"], 0.38, color=PALETTE["after"], label="Cleaned")
    ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=8.5)
    ax.set_xlabel("% missing values"); ax.set_title("Missing values per column: before vs after", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, "02_missing_before_after.png")


def plot_price(before: pd.DataFrame, after: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, d, title, col in [(axes[0], before, "Before (parsed, unvalidated)", PALETTE["before"]),
                              (axes[1], after, "After (validated + winsorised)", PALETTE["after"])]:
        cats = C.CATEGORIES
        data = [d.loc[d["category"] == c, "price_inr"].dropna().values for c in cats]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=[c.replace(" ", "\n") for c in cats], showfliers=True,
                        flierprops=dict(marker="o", markersize=3, alpha=.5))
        for p in bp["boxes"]:
            p.set_facecolor(col); p.set_alpha(.55)
        ax.set_yscale("log"); ax.set_title(title, fontweight="bold", fontsize=11); ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Price (INR, log scale)")
    fig.suptitle("Price outliers by category", fontweight="bold")
    return _save(fig, "03_price_outliers_before_after.png")


def plot_rating(before: pd.Series, after: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    b = before.dropna().round().astype(int).value_counts().sort_index()
    a = after.round().astype(int).value_counts().sort_index()
    axes[0].bar(b.index.astype(str), b.values, color=[PALETTE["before"] if (i < 1 or i > 5) else "#E8A09E" for i in b.index])
    axes[0].set_title("Raw numeric ratings (invalid values in dark red)", fontsize=10, fontweight="bold")
    axes[1].bar(a.index.astype(str), a.values, color=PALETTE["after"])
    axes[1].set_title("Cleaned ratings (1-5 only)", fontsize=10, fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Rating"); ax.set_ylabel("Reviews")
    return _save(fig, "04_rating_before_after.png")


def plot_categories(before: pd.Series, after: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    b = before.sort_values()
    axes[0].barh(b.index, b.values, color=PALETTE["before"]); axes[0].set_title(f"Raw 'category' labels ({len(b)} variants)", fontweight="bold", fontsize=10)
    a = after.sort_values()
    axes[1].barh(a.index, a.values, color=PALETTE["after"]); axes[1].set_title(f"Cleaned 'category' ({len(a)} labels)", fontweight="bold", fontsize=10)
    for ax in axes:
        ax.tick_params(axis="y", labelsize=8); ax.set_xlabel("Rows")
    return _save(fig, "05_category_before_after.png")


def plot_scorecard(score: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(score)); w = 0.36
    b = ax.bar(x - w / 2, score["before"], w, color=PALETTE["before"], label="Before")
    a = ax.bar(x + w / 2, score["after"], w, color=PALETTE["after"], label="After")
    for bars in (b, a):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{r.get_height():.0f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([d.title() for d in score["dimension"]]); ax.set_ylim(0, 112)
    ax.set_ylabel("Score (%)"); ax.set_title("Data-quality scorecard", fontweight="bold"); ax.legend(frameon=False, loc="lower right")
    return _save(fig, "06_quality_scorecard.png")


def plot_pipeline_attrition(step_log: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(step_log)), step_log["rows_out"], marker="o", color=PALETTE["accent"], lw=2)
    ax.set_xticks(range(len(step_log))); ax.set_xticklabels([s.split(" ")[0] for s in step_log["step"]])
    ax.set_xlabel("Pipeline step"); ax.set_ylabel("Rows"); ax.set_title("Row count through the pipeline", fontweight="bold")
    return _save(fig, "07_row_attrition.png")
