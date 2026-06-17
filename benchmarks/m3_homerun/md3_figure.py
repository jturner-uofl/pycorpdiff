"""Figure §13c: MD3 as the timed TRIGGER (Sethi & Kantardzic 2017), not the
misinfo detector. Top: margin-density trajectory of the cannabidiol-COVID
subcorpus vs a label-shuffle null, onset + FDA marked. Bottom: the honest caveat
— reference-window sensitivity (the magnitude is inflated by a thin early base)."""
import json
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
RED, DARK, MUTE, GRID, INK = "#b91c1c", "#7f1d1d", "#6b7280", "#e5e7eb", "#1f2937"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#d1d5db", "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "figure.dpi": 140,
})
traj = pd.read_csv(HERE / "md3_trigger.csv", parse_dates=["date"]).sort_values("date")
sens = pd.read_csv(HERE / "md3_sensitivity.csv")
M = json.load(open(HERE / "md3_meta.json"))
FDA = datetime(2020, 3, 28)

fig = plt.figure(figsize=(12.5, 8.4))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.42,
                      left=0.085, right=0.965, top=0.87, bottom=0.10)

# ---------------- top: margin-density trajectory (the trigger) ----------------
ax = fig.add_subplot(gs[0])
x, y = traj.date.values, traj.margin_density.values
ax.axhspan(0, M["ref_margin"], color=MUTE, alpha=0.06)
ax.axhline(M["ref_margin"], color=MUTE, lw=1.1, ls=":")
ax.text(x[0], M["ref_margin"], "  early Jan–Feb baseline", va="bottom", ha="left",
        fontsize=9, color=MUTE, style="italic")
ax.axhline(M["null_threshold"], color=DARK, lw=1.3, ls="--")
ax.text(x[-1], M["null_threshold"], f"label-shuffle null (95th pct) = {M['null_threshold']:.2f}  ",
        va="bottom", ha="right", fontsize=9, color=DARK, style="italic")
ax.fill_between(x, M["ref_margin"], y, color=RED, alpha=0.10)
ax.plot(x, y, color=RED, lw=2.6, solid_capstyle="round")
ax.scatter(x, y, s=42, color="white", edgecolor=RED, lw=2.0, zorder=5)
ax.axvline(FDA, color=DARK, lw=1.6)
ax.text(FDA, 1.0, "  FDA warning\n  letters Mar 28", va="top", ha="left",
        fontsize=10.5, color=DARK, fontweight="bold")
# onset callout
od = datetime.fromisoformat(M["onset_date"])
oi = int(np.argmin(np.abs(traj.date.values - np.datetime64(od))))
ax.annotate(f"onset {M['onset_week']}\n{od:%b %d} · ~1 wk before FDA",
            (traj.date.values[oi], traj.margin_density.values[oi]), xytext=(14, -34),
            textcoords="offset points", fontsize=9.5, color=INK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=MUTE, lw=1.0))
ax.text(0.015, 0.95, f"permutation p = {M['p_value']}   ·   change-type: {M['change_type']}",
        transform=ax.transAxes, va="top", fontsize=10.5, color=DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fef2f2", ec=RED, lw=1.0))
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
ax.set_ylabel("margin density\n(fraction outside the early claim-senses)", fontsize=11, fontweight="bold")
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.grid(axis="y", color=GRID, lw=0.7)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_title("MD3 as the timed trigger — WHEN the CBD–COVID discourse destabilized, not WHAT it claimed",
             fontsize=13.5, fontweight="bold", color=DARK, loc="left", pad=12)

# ---------------- bottom: the honest caveat (reference sensitivity) ----------
ax2 = fig.add_subplot(gs[1])
labels = [f"ref ≤ {c[5:]}\n(n={n:,})" for c, n in zip(sens.cut, sens.ref_n)]
colors = [RED if v > 0.5 else "#94a3b8" for v in sens.late_margin]
ax2.barh(np.arange(len(sens)), sens.late_margin, color=colors, edgecolor="white", height=0.66)
for i, v in enumerate(sens.late_margin):
    ax2.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=9, color=INK, fontweight="bold")
ax2.set_yticks(np.arange(len(sens))); ax2.set_yticklabels(labels, fontsize=8.5)
ax2.set_xlim(0, 1.05); ax2.set_xlabel("mean late-week (Apr–Jun) margin density", fontsize=10)
ax2.invert_yaxis()
for s in ["top", "right"]: ax2.spines[s].set_visible(False)
ax2.grid(axis="x", color=GRID, lw=0.7)
ax2.set_title("Honest caveat — reference-window sensitivity: let the reference see March and the "
              "margin collapses 0.84 → 0.20", fontsize=11, fontweight="bold", color=DARK, loc="left", pad=8)

fig.text(0.085, 0.018,
         "pycorpdiff · Sethi & Kantardzic (2017) margin-density drift on the cannabidiol-COVID subcorpus "
         "(PCA-50, k=4) · the drift is significant but commerce-led "
         f"({', '.join(M['drift_terms'][:5])}…), not misinfo — MD3 says WHEN/THAT; the misinfo WHAT is the "
         "targeted + LLM layer (§13a)", fontsize=7.6, color=MUTE, style="italic")
fig.savefig(HERE / "md3_trigger.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved md3_trigger.png | onset {M['onset_week']} p={M['p_value']} "
      f"peak={M['peak_margin']} | sensitivity collapse to {sens.late_margin.iloc[-1]:.2f}")
