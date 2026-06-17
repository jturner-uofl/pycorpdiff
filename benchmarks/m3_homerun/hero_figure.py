"""Hero figure A: the CBD-COVID sub-discourse life-cycle, winter-spring 2020.

Top panel  : diffusion S-curve (COVID share of all CBD discourse) with onset,
             take-off, peak, and FDA-warning markers + the lead-time bracket.
Bottom panel: claim-type composition (LLM-named clusters), graded honestly into
             explicit cure/treatment claims (dark red), health-framed/promotional
             (amber), and commercial/industry (grey). Every number is measured;
             labels are the LLM's interpretation of cited tweets, not a verdict.
"""
import json
from datetime import datetime
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyArrowPatch, Patch
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
RED, DARK, AMBER, GREY = "#b91c1c", "#7f1d1d", "#d97706", "#94a3b8"
INK, MUTE, GRID = "#1f2937", "#6b7280", "#e5e7eb"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#d1d5db", "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "figure.dpi": 140,
})
FDA = datetime(2020, 3, 28); FTC = datetime(2020, 3, 9)

traj = pd.read_csv(HERE / "diffusion.csv", parse_dates=["date"]).sort_values("date")
claims = pd.read_json(HERE / "claim_types.json").sort_values("n", ascending=False)
n_total = int(claims.n.sum())

# ---- honest onset metrics recomputed from the curve --------------------------
def lead(d): return (FDA - d).days
detect = traj[traj.covid_n >= 10]                       # first MEANINGFUL detection
first_detect = detect.iloc[0]
takeoff = traj[traj.share >= 0.005].iloc[0]             # clear take-off (>=0.5%)
peak = traj.loc[traj.share.idxmax()]

# ---- claim tiers (honest grading, not the LLM's blanket flag) ----------------
def tier(r):
    if r.is_efficacy_claim is True and r.efficacy_term_frac >= 0.30: return "explicit"
    if r.is_efficacy_claim is True: return "framed"
    return "commercial"
claims["tier"] = claims.apply(tier, axis=1)
TCOL = {"explicit": RED, "framed": AMBER, "commercial": GREY}
explicit_pct = 100 * claims[claims.tier == "explicit"].n.sum() / n_total
framed_pct = 100 * claims[claims.tier == "framed"].n.sum() / n_total

fig = plt.figure(figsize=(12.5, 9.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.95, 1.05], hspace=0.46,
                      left=0.085, right=0.965, top=0.88, bottom=0.10)

# ============================ TOP: diffusion S-curve ========================
ax = fig.add_subplot(gs[0])
x = traj.date.values; y = (traj.share * 100).values
ax.fill_between(x, 0, y, color=RED, alpha=0.10, zorder=1)
ax.plot(x, y, color=RED, lw=2.6, zorder=4, solid_capstyle="round")
ax.scatter(x, y, s=42, color="white", edgecolor=RED, lw=2.0, zorder=5)
ymax = max(y) * 1.20; ax.set_ylim(0, ymax)

ax.axvspan(FTC, FDA, color=DARK, alpha=0.05, zorder=0)
ax.axvline(FTC, color=DARK, lw=1.2, ls=":", alpha=0.8, zorder=3)
ax.axvline(FDA, color=DARK, lw=1.6, ls="-", alpha=0.9, zorder=3)
ax.text(FDA, ymax, "  FDA warning letters\n  Mar 28, 2020", va="top", ha="left",
        fontsize=10.5, color=DARK, fontweight="bold")
ax.text(FTC, ymax * 0.02, "FTC first\nsignal Mar 9 ", va="bottom", ha="right",
        fontsize=8.5, color=MUTE, style="italic")

def near(dt):
    i = int(np.argmin(np.abs(traj.date.values - np.datetime64(dt))))
    return traj.date.values[i], traj.share.values[i] * 100
for dt, lab, dx, dy in [
    (first_detect.date, f"First detectable\n{first_detect.date:%b %d} · {first_detect.covid_n} tweets", -8, 34),
    (takeoff.date, f"Take-off >0.5%\n{takeoff.date:%b %d}", -62, 30),
    (peak.date, f"Peak {peak.share*100:.1f}%\n{peak.date:%b %d}", 10, 6)]:
    px, py = near(pd.Timestamp(dt))
    ax.annotate(lab, (px, py), xytext=(dx, dy), textcoords="offset points",
                fontsize=9, color=INK, fontweight="bold", ha="center",
                arrowprops=dict(arrowstyle="-", color=MUTE, lw=0.9, alpha=0.8))

t0 = pd.Timestamp(takeoff.date); ld = lead(takeoff.date)
ytop = ymax * 0.60
arr = FancyArrowPatch((mdates.date2num(t0), ytop), (mdates.date2num(FDA), ytop),
                      arrowstyle="<->", color=DARK, lw=1.7, mutation_scale=15, zorder=6)
ax.add_patch(arr)
ax.text((mdates.date2num(t0) + mdates.date2num(FDA)) / 2, ytop,
        f"{ld} days of lead\n(~{ld/7:.0f} weeks)", ha="center", va="bottom",
        fontsize=10.5, color=DARK, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=DARK, lw=1.0, alpha=0.96))

ax.set_ylabel("COVID share of all CBD discourse", fontsize=11.5, fontweight="bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.set_xlim(traj.date.min(), traj.date.max())
ax.grid(axis="y", color=GRID, lw=0.7)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.set_title("How the CBD–COVID sub-discourse emerged — and outran the regulator",
             fontsize=15.5, fontweight="bold", color=DARK, loc="left", pad=12)

# ============================ BOTTOM: claim-types ===========================
ax2 = fig.add_subplot(gs[1])
cl = claims.head(7).iloc[::-1].reset_index(drop=True)
ypos = np.arange(len(cl))
ax2.barh(ypos, cl.pct, color=[TCOL[t] for t in cl.tier], edgecolor="white", height=0.74)
for i, (_, r) in enumerate(cl.iterrows()):
    tag = {"explicit": "  ‹!› explicit cure/treatment claim", "framed": "  · health-framed",
           "commercial": ""}[r.tier]
    ax2.text(r.pct + 0.4, i, f"{r['label']}{tag}", va="center", ha="left", fontsize=9.3,
             color=INK if r.tier == "explicit" else MUTE,
             fontweight="bold" if r.tier == "explicit" else "normal")
    ax2.text(r.pct - 0.4, i, f"{r.pct:.0f}%", va="center", ha="right", fontsize=8.5,
             color="white" if r.pct > 4 else MUTE, fontweight="bold")
ax2.set_yticks([]); ax2.set_xlim(0, cl.pct.max() * 2.5)
ax2.set_xlabel("share of cannabidiol–COVID tweets", fontsize=10)
for s in ["top", "right", "left"]: ax2.spines[s].set_visible(False)
ax2.grid(axis="x", color=GRID, lw=0.7)
ax2.legend(handles=[Patch(fc=RED, label=f"explicit cure/treatment ({explicit_pct:.0f}%)"),
                    Patch(fc=AMBER, label=f"health-framed / promotional ({framed_pct:.0f}%)"),
                    Patch(fc=GREY, label="commercial / industry")],
           loc="lower right", frameon=False, fontsize=8.5, handlelength=1.1)
ax2.set_title(f"What the {n_total:,} tweets were claiming  —  graded by claim strength",
              fontsize=12.5, fontweight="bold", color=DARK, loc="left", pad=10)

fig.text(0.085, 0.018,
         "pycorpdiff · 3.46M CBD tweets (2019–2020), cannabidiol-sense filtered · "
         "share = COVID-mentioning ÷ all CBD tweets per week (robust to batch sampling) · "
         "cluster labels = LLM reading of cited tweets, never a veracity verdict",
         fontsize=7.8, color=MUTE, style="italic")

fig.savefig(HERE / "cbd_covid_lifecycle.png", dpi=200, bbox_inches="tight", facecolor="white")
print(f"saved cbd_covid_lifecycle.png | explicit={explicit_pct:.0f}% framed={framed_pct:.0f}% "
      f"| take-off {takeoff.date:%b %d} lead {lead(takeoff.date)}d | peak {peak.share*100:.1f}%")
