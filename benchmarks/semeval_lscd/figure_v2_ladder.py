"""The v2 narrative figure (CORRECTED): the SemEval saturation was a Mahalanobis
artifact, fixable two ways. rho ladder on SemEval English LSCD. Saves
v2_ladder.png. Numbers from the cap-300 apples-to-apples runs; the Mahalanobis
default is the documented original (saturated, no ranking signal)."""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
ladder = [
    ("margin · Mahalanobis novelty (pcd default)", -0.19),   # saturated -> no signal
    ("mean-shift / cosine (raw)",                   0.26),
    ("MMD + background-correction (M2)",            0.27),
    ("mean-shift + background-correction (M1)",     0.33),
    ("margin · Euclidean novelty (metric fix)",     0.33),
]
names = [n for n, _ in ladder]; vals = [v for _, v in ladder]
colors = ["#c0392b" if v < 0.10 else "#d99020" if v < 0.30 else "#1e8e4e" for v in vals]

fig, ax = plt.subplots(figsize=(9.6, 4.6))
y = list(range(len(names)))
ax.barh(y, vals, color=colors, height=0.62)
ax.axvline(0.422, ls="--", color="#555", lw=1.3)
ax.text(0.422, -0.8, "SemEval-2020 best ≈ 0.42", fontsize=8.5, color="#555", ha="center")
ax.axvline(0, color="k", lw=0.9)
ax.text(-0.19, 0, "  saturated", va="center", ha="left", fontsize=8, color="#7a1f15", style="italic")
for i, v in enumerate(vals):
    off = 0.012 if v >= 0 else -0.012
    ax.text(v + off, i, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right",
            fontsize=10, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9.5)
ax.invert_yaxis(); ax.set_xlim(-0.30, 0.56)
ax.set_xlabel("Spearman ρ vs graded gold  (SemEval-2020 English LSCD — used only as a ruler)")
ax.set_title("The SemEval saturation was a metric artifact, not a design flaw —\n"
             "two cheap fixes recover competitive signal", fontsize=11.5)
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig(ROOT/"v2_ladder.png", dpi=140)
print("saved v2_ladder.png")
