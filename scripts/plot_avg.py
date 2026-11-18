# Plots averaged.csv and cross_q.csv written by analyze_all.py.
# Usage: python plot_avg.py results/analysis
import sys, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def load(name):
    rows = [r for r in csv.DictReader(l for l in open(os.path.join(sys.argv[1], name)) if not l.startswith("#"))]
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
A, C = load("averaged.csv"), load("cross_q.csv")
Ts = sorted(set(A["T"])); doses = sorted(set(A["nvac"].astype(int)))
cols = ["#1f4e79", "#2f6fb0", "#6aa0d6", "#e8935a", "#c8641f", "#8a3b0f"]
PAPER = "--paper" in sys.argv          # page width layout: 2 by 2 panels, 6.5 in wide, 300 dpi
plt.rcParams.update({"font.size": 10 if PAPER else 15})
if PAPER:
    fig, ax = plt.subplots(2, 2, figsize=(6.5, 5.6), constrained_layout=True); ax = ax.ravel()
else:
    fig, ax = plt.subplots(1, 4, figsize=(22, 5.6), constrained_layout=True)
for c, t in zip(cols, Ts):
    m = A["T"] == t; mc = C["T"] == t
    ax[0].plot(A["nvac"][m], A["Hnorm_of_avg_map"][m], marker="o", color=c, lw=2, label=f"T = {t:g} J")
    ax[1].plot(A["nvac"][m], A["peak_of_avg_map"][m], marker="o", color=c, lw=2)
    ax[1].plot(A["nvac"][m], A["mean_of_config_peaks"][m], marker="s", mfc="none", ls="--", color=c, lw=1.3)
    ax[2].plot(A["nvac"][m], A["mean_f_over_q"][m], marker="o", color=c, lw=2)
    ax[3].errorbar(C["nvac"][mc], C["cross_q_share"][mc], yerr=C["cross_q_sem"][mc], marker="o", capsize=3, color=c, lw=2)
ax[2].axhline(1.0, color="k", ls="--", lw=1.5)
ax[3].set_ylim(0, None)
ax[0].set_title("Spread of averaged map"); ax[0].set_ylabel("H_norm")
ax[1].set_title("Peak f: averaged (solid),\nper config (dashed)" if PAPER else "Peak f: averaged (solid), per config (dashed)"); ax[1].set_ylabel("max over q")
ax[2].set_title("Mean f over q; g sum rule (dashed)"); ax[2].set_ylabel("mean over q")
ax[3].set_title("QFI weight linking different q"); ax[3].set_ylabel("cross q share")
for a in ax:
    a.set_xticks(doses); a.set_xlabel("vacancies"); a.spines[["top", "right"]].set_visible(False)
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=len(Ts), frameon=False, fontsize=9 if PAPER else 13)
fig.savefig("avg_dose_paper.png" if PAPER else "avg_dose.png", dpi=300 if PAPER else 130, facecolor="#ffffff")
