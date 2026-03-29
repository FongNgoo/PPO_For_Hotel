# evaluation/visualize.py
"""
Visualization Module for Multi-Room Hotel Pricing Evaluation.

Generates comprehensive charts for comparing PPO against baseline strategies:
1. Revenue comparison (bar chart with CI error bars)
2. Revenue distribution (box plot)
3. Revenue lift vs baseline (horizontal bar)
4. Revenue per booking comparison
5. Retention rate comparison
6. Price stability (variance) per room
7. Price deviation from ADR reference
8. Statistical significance heatmap
9. Cumulative revenue over episodes
10. Per-room pricing behavior radar / grouped bar

All figures are saved as high-resolution PNG files.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional
from datetime import datetime

# ── internal imports ────────────────────────────────────────────────────────
# Allow running from project root
import sys
sys.path.insert(0, '.')

# These are imported lazily inside functions to avoid circular imports,
# but we reference their types via strings in annotations.
# ────────────────────────────────────────────────────────────────────────────


# ============================================================
# STYLE CONSTANTS
# ============================================================

# Colour palette — PPO first, then baselines
STRATEGY_COLORS = {
    "PPO (Ours)":           "#2196F3",   # blue
    "Fixed (α=1.0)":        "#9E9E9E",   # grey  ← primary baseline
    "Fixed (α=0.9)":        "#BDBDBD",   # light grey
    "Fixed (α=1.1)":        "#757575",   # dark grey
    "Segmented Pricing":    "#FF9800",   # orange
    "Weekend/Weekday":      "#4CAF50",   # green
    "Demand Heuristic":     "#9C27B0",   # purple
    "Rule-Based Dynamic":   "#F44336",   # red
    "Random":               "#00BCD4",   # cyan
}

DEFAULT_COLOR = "#607D8B"   # blue-grey for unknown strategies

FIGURE_DPI   = 150
FIGURE_STYLE = "seaborn-v0_8-whitegrid"


def _color(name: str) -> str:
    return STRATEGY_COLORS.get(name, DEFAULT_COLOR)


def _savefig(fig: plt.Figure, path: str) -> None:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {path}")


def _strategy_order(names: List[str]) -> List[str]:
    """Sort: PPO first, then fixed baselines, then others, Random last."""
    priority = {
        "PPO (Ours)":        0,
        "Fixed (α=1.0)":     1,
        "Fixed (α=0.9)":     2,
        "Fixed (α=1.1)":     3,
        "Segmented Pricing": 4,
        "Weekend/Weekday":   5,
        "Demand Heuristic":  6,
        "Rule-Based Dynamic":7,
        "Random":            8,
    }
    return sorted(names, key=lambda n: priority.get(n, 99))


# ============================================================
# 1. REVENUE COMPARISON BAR CHART (with 95 % CI)
# ============================================================

def plot_revenue_comparison(metrics: dict, output_dir: str) -> str:
    """Bar chart of mean revenue per strategy with 95 % confidence intervals."""

    ordered = _strategy_order(list(metrics.keys()))
    names   = ordered
    means   = [metrics[n].mean_revenue     for n in names]
    ci_lo   = [metrics[n].revenue_ci_lower for n in names]
    ci_hi   = [metrics[n].revenue_ci_upper for n in names]
    colors  = [_color(n)                   for n in names]

    err_lo = [m - lo for m, lo in zip(means, ci_lo)]
    err_hi = [hi - m for m, hi in zip(means, ci_hi)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    bars = ax.bar(x, means, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)

    ax.errorbar(x, means,
                yerr=[err_lo, err_hi],
                fmt="none", color="black",
                capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

    # Value labels on top of bars
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(err_hi) * 0.05,
                f"€{mean:,.0f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Total Revenue (€)", fontsize=11)
    ax.set_title("Revenue Comparison Across Pricing Strategies\n(with 95% Confidence Intervals)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.set_ylim(0, max(means) * 1.18)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "01_revenue_comparison.png")
    _savefig(fig, path)
    return path


# ============================================================
# 2. REVENUE DISTRIBUTION BOX PLOT
# ============================================================

def plot_revenue_distribution(metrics: dict, output_dir: str) -> str:
    """Box plot showing revenue distribution across episodes per strategy."""

    ordered = _strategy_order(list(metrics.keys()))
    data    = [metrics[n].all_revenues for n in ordered]
    colors  = [_color(n)               for n in ordered]

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticks(range(1, len(ordered) + 1))
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Total Revenue per Episode (€)", fontsize=11)
    ax.set_title("Revenue Distribution per Strategy\n(Box = IQR, Whiskers = 1.5×IQR)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax.grid(axis="y", alpha=0.4)

    fig.tight_layout()
    path = os.path.join(output_dir, "02_revenue_distribution.png")
    _savefig(fig, path)
    return path


# ============================================================
# 3. REVENUE LIFT VS BASELINE (horizontal bar)
# ============================================================

def plot_revenue_lift(statistical_tests: dict, output_dir: str) -> str:
    """Horizontal bar chart of revenue lift (%) vs Fixed (α=1.0) baseline."""

    if not statistical_tests:
        print("  ⚠ No statistical tests available — skipping lift chart.")
        return ""

    ordered = _strategy_order(list(statistical_tests.keys()))
    lifts   = [statistical_tests[n].revenue_lift_pct for n in ordered]
    sigs    = [statistical_tests[n].revenue_significant for n in ordered]
    pvals   = [statistical_tests[n].revenue_p_value    for n in ordered]

    colors = []
    for lift, sig in zip(lifts, sigs):
        if sig and lift > 0:
            colors.append("#2196F3")   # significantly better → blue
        elif sig and lift < 0:
            colors.append("#F44336")   # significantly worse  → red
        else:
            colors.append("#9E9E9E")   # not significant      → grey

    fig, ax = plt.subplots(figsize=(10, max(5, len(ordered) * 0.7)))
    y = np.arange(len(ordered))

    bars = ax.barh(y, lifts, color=colors, height=0.55,
                   edgecolor="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")

    # Annotate with lift % and p-value
    for i, (bar, lift, sig, pval) in enumerate(zip(bars, lifts, sigs, pvals)):
        label = f"{lift:+.1f}%  (p={pval:.3f}{'*' if sig else ''})"
        x_pos = lift + (0.3 if lift >= 0 else -0.3)
        ha    = "left" if lift >= 0 else "right"
        ax.text(x_pos, i, label, va="center", ha=ha, fontsize=8.5)

    ax.set_yticks(y)
    ax.set_yticklabels(ordered, fontsize=9)
    ax.set_xlabel("Revenue Lift vs Fixed (α=1.0) Baseline (%)", fontsize=11)
    ax.set_title("Revenue Lift Comparison\n(* = statistically significant at α=0.05)",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)

    # Legend
    legend_patches = [
        mpatches.Patch(color="#2196F3", label="Significantly better"),
        mpatches.Patch(color="#F44336", label="Significantly worse"),
        mpatches.Patch(color="#9E9E9E", label="Not significant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "03_revenue_lift.png")
    _savefig(fig, path)
    return path


# ============================================================
# 4. REVENUE PER BOOKING
# ============================================================

def plot_revenue_per_booking(metrics: dict, output_dir: str) -> str:
    """Bar chart of Expected Revenue per Booking (€/booking)."""

    ordered = _strategy_order(list(metrics.keys()))
    rpb     = [metrics[n].revenue_per_booking for n in ordered]
    colors  = [_color(n)                       for n in ordered]

    fig, ax = plt.subplots(figsize=(12, 5))
    x    = np.arange(len(ordered))
    bars = ax.bar(x, rpb, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, rpb):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rpb) * 0.01,
                f"€{val:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Revenue per Booking (€)", fontsize=11)
    ax.set_title("Expected Revenue per Booking by Strategy", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:.0f}"))
    ax.set_ylim(0, max(rpb) * 1.15)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "04_revenue_per_booking.png")
    _savefig(fig, path)
    return path


# ============================================================
# 5. RETENTION RATE
# ============================================================

def plot_retention_rate(metrics: dict, output_dir: str) -> str:
    """Bar chart of expected booking retention rate (bookings/day)."""

    ordered    = _strategy_order(list(metrics.keys()))
    retentions = [metrics[n].daily_retention_rate for n in ordered]
    colors     = [_color(n)                        for n in ordered]

    fig, ax = plt.subplots(figsize=(12, 5))
    x    = np.arange(len(ordered))
    bars = ax.bar(x, retentions, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, retentions):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(retentions) * 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Bookings per Day", fontsize=11)
    ax.set_title("Expected Retention Rate (Bookings / Day) by Strategy",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(retentions) * 1.18)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "05_retention_rate.png")
    _savefig(fig, path)
    return path


# ============================================================
# 6. PRICE DEVIATION FROM ADR REFERENCE (per room)
# ============================================================

def plot_price_deviation(metrics: dict, room_types: List[str],
                         output_dir: str) -> str:
    """
    Grouped bar chart showing mean price deviation (α - 1.0) per room type
    for each strategy.
    """

    ordered = _strategy_order(list(metrics.keys()))
    n_rooms = len(room_types)
    n_strats = len(ordered)

    fig, ax = plt.subplots(figsize=(max(10, n_rooms * 2.5), 6))

    bar_width = 0.8 / n_strats
    x = np.arange(n_rooms)

    for i, name in enumerate(ordered):
        rm = metrics[name].room_metrics
        deviations = [rm[room]["price_deviation"] for room in room_types]
        offset = (i - n_strats / 2 + 0.5) * bar_width
        ax.bar(x + offset, deviations, bar_width,
               label=name, color=_color(name),
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(room_types, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean α Deviation from 1.0 (i.e. from ADR reference)", fontsize=10)
    ax.set_title("Price Deviation from ADR Reference per Room Type",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "06_price_deviation.png")
    _savefig(fig, path)
    return path


# ============================================================
# 7. PRICE VARIANCE (stability) per room
# ============================================================

def plot_price_variance(metrics: dict, room_types: List[str],
                        output_dir: str) -> str:
    """
    Grouped bar chart of α variance per room — lower = more stable pricing.
    """

    ordered  = _strategy_order(list(metrics.keys()))
    n_rooms  = len(room_types)
    n_strats = len(ordered)

    fig, ax = plt.subplots(figsize=(max(10, n_rooms * 2.5), 6))

    bar_width = 0.8 / n_strats
    x = np.arange(n_rooms)

    for i, name in enumerate(ordered):
        rm = metrics[name].room_metrics
        variances = [rm[room]["alpha_variance"] for room in room_types]
        offset = (i - n_strats / 2 + 0.5) * bar_width
        ax.bar(x + offset, variances, bar_width,
               label=name, color=_color(name),
               edgecolor="white", linewidth=0.5, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(room_types, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("α Variance (lower = more stable)", fontsize=10)
    ax.set_title("Price Stability (α Variance) per Room Type",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "07_price_variance.png")
    _savefig(fig, path)
    return path


# ============================================================
# 8. STATISTICAL SIGNIFICANCE HEATMAP
# ============================================================

def plot_significance_heatmap(statistical_tests: dict, output_dir: str) -> str:
    """
    Heatmap of p-values for each strategy vs baseline.
    Cells coloured by significance level.
    """

    if not statistical_tests:
        print("  ⚠ No statistical tests — skipping heatmap.")
        return ""

    ordered  = _strategy_order(list(statistical_tests.keys()))
    metrics  = ["Revenue p-value", "Booking p-value"]
    n_s, n_m = len(ordered), len(metrics)

    matrix = np.zeros((n_s, n_m))
    for i, name in enumerate(ordered):
        t = statistical_tests[name]
        matrix[i, 0] = t.revenue_p_value
        matrix[i, 1] = t.booking_p_value

    fig, ax = plt.subplots(figsize=(6, max(4, n_s * 0.7)))

    im = ax.imshow(matrix, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=0.1)

    ax.set_xticks(range(n_m))
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks(range(n_s))
    ax.set_yticklabels(ordered, fontsize=9)

    # Annotate cells
    for i in range(n_s):
        for j in range(n_m):
            val = matrix[i, j]
            text_color = "white" if val < 0.03 else "black"
            marker = "*" if val < 0.05 else ""
            ax.text(j, i, f"{val:.3f}{marker}",
                    ha="center", va="center", fontsize=9,
                    color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    ax.set_title("Statistical Significance Heatmap\n(vs Fixed α=1.0 baseline, * p<0.05)",
                 fontsize=12, fontweight="bold")

    fig.tight_layout()
    path = os.path.join(output_dir, "08_significance_heatmap.png")
    _savefig(fig, path)
    return path


# ============================================================
# 9. EFFECT SIZE (Cohen's d) BAR CHART
# ============================================================

def plot_effect_size(statistical_tests: dict, output_dir: str) -> str:
    """Bar chart of Cohen's d effect sizes."""

    if not statistical_tests:
        return ""

    ordered = _strategy_order(list(statistical_tests.keys()))
    ds      = [statistical_tests[n].revenue_effect_size for n in ordered]
    colors  = [_color(n) for n in ordered]

    fig, ax = plt.subplots(figsize=(10, 5))
    x    = np.arange(len(ordered))
    bars = ax.bar(x, ds, color=colors, width=0.6, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.axhline(0,    color="black", linewidth=1,   linestyle="--")
    ax.axhline(0.2,  color="green", linewidth=0.8, linestyle=":",  label="|d|=0.2 (small)")
    ax.axhline(0.5,  color="orange",linewidth=0.8, linestyle=":",  label="|d|=0.5 (medium)")
    ax.axhline(0.8,  color="red",   linewidth=0.8, linestyle=":",  label="|d|=0.8 (large)")
    ax.axhline(-0.2, color="green", linewidth=0.8, linestyle=":")
    ax.axhline(-0.5, color="orange",linewidth=0.8, linestyle=":")
    ax.axhline(-0.8, color="red",   linewidth=0.8, linestyle=":")

    for bar, val in zip(bars, ds):
        y_pos = val + 0.02 * (1 if val >= 0 else -1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                y_pos, f"{val:.2f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's d Effect Size", fontsize=11)
    ax.set_title("Effect Size vs Fixed (α=1.0) Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "09_effect_size.png")
    _savefig(fig, path)
    return path


# ============================================================
# 10. MEAN PRICE PER ROOM (absolute €)
# ============================================================

def plot_mean_price_per_room(metrics: dict, room_types: List[str],
                             adr_refs: Dict[str, float],
                             output_dir: str) -> str:
    """
    Grouped bar chart of mean suggested price per room type.
    Reference ADR shown as a horizontal line per room.
    """

    ordered  = _strategy_order(list(metrics.keys()))
    n_rooms  = len(room_types)
    n_strats = len(ordered)

    fig, ax = plt.subplots(figsize=(max(10, n_rooms * 2.5), 6))

    bar_width = 0.8 / n_strats
    x = np.arange(n_rooms)

    for i, name in enumerate(ordered):
        rm = metrics[name].room_metrics
        prices = [rm[room]["mean_price"] for room in room_types]
        offset = (i - n_strats / 2 + 0.5) * bar_width
        ax.bar(x + offset, prices, bar_width,
               label=name, color=_color(name),
               edgecolor="white", linewidth=0.5, zorder=3)

    # ADR reference markers
    for j, room in enumerate(room_types):
        ax.hlines(adr_refs[room], j - 0.4, j + 0.4,
                  colors="black", linewidths=1.5, linestyles="--", zorder=5)

    # Dummy entry for legend
    ax.plot([], [], "k--", linewidth=1.5, label="ADR Reference")

    ax.set_xticks(x)
    ax.set_xticklabels(room_types, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean Suggested Price (€)", fontsize=10)
    ax.set_title("Mean Suggested Price per Room Type\n(dashed line = ADR reference)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:.0f}"))
    ax.grid(axis="y", alpha=0.4, zorder=0)

    fig.tight_layout()
    path = os.path.join(output_dir, "10_mean_price_per_room.png")
    _savefig(fig, path)
    return path


# ============================================================
# 11. DASHBOARD: 4-PANEL SUMMARY
# ============================================================

def plot_dashboard(metrics: dict, statistical_tests: dict,
                   room_types: List[str], output_dir: str) -> str:
    """
    4-panel summary dashboard:
    Top-left:  Revenue comparison bar
    Top-right: Revenue lift horizontal bar
    Bot-left:  Revenue per booking
    Bot-right: Retention rate
    """

    ordered = _strategy_order(list(metrics.keys()))
    colors  = [_color(n) for n in ordered]

    fig = plt.figure(figsize=(18, 11))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: Revenue comparison ──────────────────────────
    ax1  = fig.add_subplot(gs[0, 0])
    means = [metrics[n].mean_revenue for n in ordered]
    err_lo = [metrics[n].mean_revenue - metrics[n].revenue_ci_lower for n in ordered]
    err_hi = [metrics[n].revenue_ci_upper - metrics[n].mean_revenue for n in ordered]
    x1 = np.arange(len(ordered))
    ax1.bar(x1, means, color=colors, width=0.6, zorder=3,
            edgecolor="white", linewidth=0.6)
    ax1.errorbar(x1, means, yerr=[err_lo, err_hi],
                 fmt="none", color="black", capsize=4, linewidth=1.3, zorder=4)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(ordered, rotation=30, ha="right", fontsize=7.5)
    ax1.set_ylabel("Mean Revenue (€)", fontsize=9)
    ax1.set_title("(a) Revenue Comparison (95% CI)", fontsize=10, fontweight="bold")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:,.0f}"))
    ax1.grid(axis="y", alpha=0.35, zorder=0)

    # ── Panel 2: Revenue lift ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    if statistical_tests:
        lift_names  = _strategy_order(list(statistical_tests.keys()))
        lifts       = [statistical_tests[n].revenue_lift_pct for n in lift_names]
        sigs        = [statistical_tests[n].revenue_significant for n in lift_names]
        lift_colors = ["#2196F3" if (s and l > 0) else "#F44336" if (s and l < 0)
                       else "#9E9E9E" for l, s in zip(lifts, sigs)]
        y2 = np.arange(len(lift_names))
        ax2.barh(y2, lifts, color=lift_colors, height=0.55, zorder=3,
                 edgecolor="white", linewidth=0.6)
        ax2.axvline(0, color="black", linewidth=1.1, linestyle="--")
        ax2.set_yticks(y2)
        ax2.set_yticklabels(lift_names, fontsize=7.5)
        ax2.set_xlabel("Revenue Lift (%)", fontsize=9)
        ax2.set_title("(b) Revenue Lift vs Fixed α=1.0", fontsize=10, fontweight="bold")
        ax2.grid(axis="x", alpha=0.35, zorder=0)
    else:
        ax2.text(0.5, 0.5, "No statistical tests", ha="center", va="center",
                 transform=ax2.transAxes, fontsize=10, color="grey")
        ax2.set_title("(b) Revenue Lift", fontsize=10, fontweight="bold")

    # ── Panel 3: Rev/Booking ──────────────────────────────────
    ax3  = fig.add_subplot(gs[1, 0])
    rpbs = [metrics[n].revenue_per_booking for n in ordered]
    x3   = np.arange(len(ordered))
    ax3.bar(x3, rpbs, color=colors, width=0.6, zorder=3,
            edgecolor="white", linewidth=0.6)
    ax3.set_xticks(x3)
    ax3.set_xticklabels(ordered, rotation=30, ha="right", fontsize=7.5)
    ax3.set_ylabel("Revenue per Booking (€)", fontsize=9)
    ax3.set_title("(c) Expected Revenue per Booking", fontsize=10, fontweight="bold")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"€{v:.0f}"))
    ax3.grid(axis="y", alpha=0.35, zorder=0)

    # ── Panel 4: Retention rate ───────────────────────────────
    ax4  = fig.add_subplot(gs[1, 1])
    rets = [metrics[n].daily_retention_rate for n in ordered]
    x4   = np.arange(len(ordered))
    ax4.bar(x4, rets, color=colors, width=0.6, zorder=3,
            edgecolor="white", linewidth=0.6)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(ordered, rotation=30, ha="right", fontsize=7.5)
    ax4.set_ylabel("Bookings / Day", fontsize=9)
    ax4.set_title("(d) Expected Retention Rate", fontsize=10, fontweight="bold")
    ax4.grid(axis="y", alpha=0.35, zorder=0)

    fig.suptitle("Multi-Room Hotel Pricing — Evaluation Dashboard",
                 fontsize=14, fontweight="bold", y=1.01)

    path = os.path.join(output_dir, "00_dashboard.png")
    _savefig(fig, path)
    return path


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def generate_all_visualizations(
    metrics: dict,
    statistical_tests: dict,
    room_types: List[str],
    adr_refs: Dict[str, float],
    output_dir: str = "evaluation_results"
) -> List[str]:
    """
    Generate and save all evaluation plots.

    Parameters
    ----------
    metrics           : Dict[str, StrategyMetrics]  from ComprehensiveEvaluator
    statistical_tests : Dict[str, StatisticalTest]  from run_statistical_analysis
    room_types        : list of room type strings
    adr_refs          : dict mapping room_type → reference ADR (€)
    output_dir        : directory to save PNG files

    Returns
    -------
    List of saved file paths.
    """

    try:
        plt.style.use(FIGURE_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")   # fallback for older matplotlib

    os.makedirs(output_dir, exist_ok=True)
    saved = []

    print(f"\n  Generating visualizations → {output_dir}/")

    tasks = [
        ("Dashboard (4-panel)",        lambda: plot_dashboard(metrics, statistical_tests, room_types, output_dir)),
        ("Revenue comparison",         lambda: plot_revenue_comparison(metrics, output_dir)),
        ("Revenue distribution",       lambda: plot_revenue_distribution(metrics, output_dir)),
        ("Revenue lift",               lambda: plot_revenue_lift(statistical_tests, output_dir)),
        ("Revenue per booking",        lambda: plot_revenue_per_booking(metrics, output_dir)),
        ("Retention rate",             lambda: plot_retention_rate(metrics, output_dir)),
        ("Price deviation per room",   lambda: plot_price_deviation(metrics, room_types, output_dir)),
        ("Price variance per room",    lambda: plot_price_variance(metrics, room_types, output_dir)),
        ("Significance heatmap",       lambda: plot_significance_heatmap(statistical_tests, output_dir)),
        ("Effect size (Cohen's d)",    lambda: plot_effect_size(statistical_tests, output_dir)),
        ("Mean price per room",        lambda: plot_mean_price_per_room(metrics, room_types, adr_refs, output_dir)),
    ]

    for label, fn in tasks:
        try:
            path = fn()
            if path:
                saved.append(path)
        except Exception as exc:
            print(f"  ⚠ '{label}' failed: {exc}")

    print(f"\n  ✓ {len(saved)} visualizations saved.")
    return saved
