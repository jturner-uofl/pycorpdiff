"""Hero figure B: the CBD medical-claim landscape.

Each condition is a bubble: x = efficacy-claim volume (log), y = claim stridency
(share of mentions using absolute cure/treat/prevent language), size = total
mentions, colour = veracity STATUS from the external record (green approved /
amber emerging / red warned-or-unproven). COVID is one bubble among many;
epilepsy is the green reversal sitting inside the red danger zone.
"""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = Path(__file__).parent
GREEN, AMBER, RED = "#15803d", "#d97706", "#b91c1c"
DARK, INK, MUTE, GRID = "#7f1d1d", "#1f2937", "#6b7280", "#e5e7eb"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#d1d5db", "axes.linewidth": 0.8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": MUTE, "ytick.color": MUTE, "figure.dpi": 140,
})
land = pd.read_csv(HERE / "landscape.csv")

def band(s):
    if "approved" in s: return "approved"
    if "warned" in s or s.startswith("Unproven"): return "unproven"
    return "emerging"
land["band"] = land.status.apply(band)
BCOL = {"approved": GREEN, "emerging": AMBER, "unproven": RED}
land["y"] = land.eff_hard_frac * 100
land["x"] = land.eff_hard

fig, ax = plt.subplots(figsize=(12.5, 8.2))
fig.subplots_adjust(left=0.085, right=0.965, top=0.86, bottom=0.10)

# stridency-vs-evidence guide band (upper region = absolute-cure language)
ax.axhspan(40, 100, color=RED, alpha=0.04, zorder=0)
ax.text(land.x.max() * 1.6, 52, "high-stridency zone\n(absolute “cure” language)",
        fontsize=8.5, color=RED, ha="right", style="italic", alpha=0.8)

sizes = 60 + (land.n_mentions / land.n_mentions.max()) * 2600
for _, r in land.iterrows():
    ax.scatter(r.x, r.y, s=sizes[r.name] if r.name in sizes.index else 200,
               color=BCOL[r.band], alpha=0.78, edgecolor="white", lw=1.4, zorder=4)

# labels with light leader offsets
NUDGE = {  # per-condition (dx_pts, dy_pts) to reduce overlap
    "Cancer": (0, 20), "Epilepsy / seizures": (0, -26), "Chronic pain": (0, -24),
    "Anxiety": (0, 20), "Alzheimer's / dementia": (-6, 16), "COVID-19": (8, -20),
    "Inflammation": (40, 6), "Addiction / opioid": (44, 4), "Depression": (-40, 8),
    "PTSD": (-36, -10), "Autism": (40, 8), "Diabetes": (40, -6), "Sleep / insomnia": (44, 0),
}
for _, r in land.iterrows():
    dx, dy = NUDGE.get(r.condition, (0, 16))
    ax.annotate(f"{r.condition}\n{r.peak_year}", (r.x, r.y), xytext=(dx, dy),
                textcoords="offset points", ha="center", va="center", fontsize=8.6,
                color=INK, fontweight="bold" if r.band != "emerging" else "normal",
                linespacing=1.0)

# epilepsy reversal callout
ep = land[land.condition == "Epilepsy / seizures"].iloc[0]
ax.annotate("THE REVERSAL: looked like misinfo in 2015,\nFDA-approved (Epidiolex) in 2018",
            (ep.x, ep.y), xytext=(ep.x * 0.30, 47), fontsize=9.2, color=GREEN, fontweight="bold",
            ha="center", arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.35", fc="#f0fdf4", ec=GREEN, lw=1.1))
# covid showcase link
cv = land[land.condition == "COVID-19"].iloc[0]
ax.annotate("the Movement-A showcase\n(FDA-warned Mar 2020)", (cv.x, cv.y),
            xytext=(cv.x * 2.4, 12), fontsize=9, color=RED, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.4),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fef2f2", ec=RED, lw=1.0))

ax.set_xscale("log")
ax.set_xlim(land.x.min() * 0.55, land.x.max() * 2.3)
ax.set_ylim(12, 66)
ax.set_xlabel("efficacy-claim tweets  (absolute cure/treat/prevent language, log scale)",
              fontsize=11, fontweight="bold")
ax.set_ylabel("claim stridency\n(% of mentions making an absolute cure claim)",
              fontsize=11, fontweight="bold")
ax.grid(True, which="both", color=GRID, lw=0.6, alpha=0.7)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(handles=[Patch(fc=GREEN, label="FDA-approved (epilepsy)"),
                   Patch(fc=AMBER, label="emerging / under study"),
                   Patch(fc=RED, label="FDA-warned or unproven")],
          loc="upper left", frameon=False, fontsize=9.5, title="veracity status (external record)",
          title_fontsize=9.5)
ax.set_title("The CBD medical-claim landscape — COVID is one sub-discourse among many",
             fontsize=15.5, fontweight="bold", color=DARK, loc="left", pad=14)
fig.text(0.085, 0.022,
         f"pycorpdiff · {int(land.n_mentions.sum()):,} condition-mentions across 3.46M CBD tweets (2011–2021) · "
         "bubble size = total mentions · status assigned from FDA/clinical record, never from the model · "
         "the most absolute “cure” language sits on unproven conditions (Alzheimer’s, autism, cancer) — "
         "epilepsy carried the same stridency in 2015, yet was FDA-approved in 2018",
         fontsize=7.6, color=MUTE, style="italic")
fig.savefig(HERE / "cbd_claim_landscape.png", dpi=200, bbox_inches="tight", facecolor="white")
print("saved cbd_claim_landscape.png")
