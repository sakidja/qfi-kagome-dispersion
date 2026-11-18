#!/usr/bin/env python3
"""
Bond map of a kagome_gs.py ground state result (27 sites and other clusters).

For every result file in the folder it draws two views of the same data:
   left   isolated view: the cluster centred on the vacancy, to follow the chain away from it
   right  tiled view: the cluster repeated over the plane (dots mark the one computed copy),
          so every triangle closes and the periodic images of the vacancy are visible
Bond colour = singlet fraction p_s = 1/4 - 3G/4 (blue above 1/2: the two spins are entangled as a
pair; red below). Bond width = |G|. Cross = vacancy.

Usage: python gs_map.py gs        (one hole runs)
       python gs_map.py gs2       (two hole runs)
Output: <dir>/<file>_map.png at 300 dpi.
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from qfi_kagome_ed import kagome_cluster

d_ = sys.argv[1] if len(sys.argv) > 1 else "gs"
norm = TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=1.0)
for p in sorted(glob.glob(os.path.join(d_, "*.npz"))):
    d = np.load(p)
    L1, L2 = [int(x) for x in d["L"]]
    G = d["G"]; vac = [int(v) for v in d["vac"]]; keep = [int(k) for k in d["keep"]]; new = {s: k for k, s in enumerate(keep)}
    pos0, bonds0, A = kagome_cluster(L1, L2)
    T1, T2 = L1 * A[0], L2 * A[1]
    centre = pos0[vac].mean(axis=0) if len(vac) == 1 else (pos0[vac[0]] if vac else pos0.mean(axis=0))

    def nearest(r, ref):
        n = min(((n1, n2) for n1 in range(-2, 3) for n2 in range(-2, 3)),
                key=lambda n: np.linalg.norm(r + n[0] * T1 + n[1] * T2 - ref))
        return r + n[0] * T1 + n[1] * T2

    pos = np.array([nearest(r, centre) for r in pos0])
    live = [(i, j) for i, j in bonds0 if i in new and j in new]
    fig, ax = plt.subplots(1, 2, figsize=(6.5, 3.6), constrained_layout=True)
    plt.rcParams.update({"font.size": 9})
    # isolated view
    segs, val = [], []
    for i, j in live:
        g = G[new[i], new[j]]; pj = nearest(pos0[j], pos[i])
        if np.allclose(pj, pos[j]):
            segs.append([pos[i], pos[j]]); val.append(g)
        else:
            pi = nearest(pos0[i], pos[j])
            segs.append([pos[i], 0.5 * (pos[i] + pj)]); val.append(g)
            segs.append([pos[j], 0.5 * (pos[j] + pi)]); val.append(g)
    val = np.array(val)
    lc = LineCollection(segs, cmap="RdBu", norm=norm, linewidths=1.0 + 5.0 * np.abs(val), capstyle="round")
    lc.set_array(0.25 - 0.75 * val); ax[0].add_collection(lc)
    ax[0].scatter(pos[keep, 0], pos[keep, 1], s=10, c="k", zorder=3)
    for v in vac:
        ax[0].scatter(*pos[v], marker="x", s=70, c="k", linewidths=2, zorder=4)
    ax[0].set_title("Isolated view")
    # tiled view
    for n1 in range(-3, 4):
        for n2 in range(-3, 4):
            sh = n1 * T1 + n2 * T2; segs, val = [], []
            for i, j in live:
                pj = nearest(pos0[j], pos[i])
                segs.append([pos[i] + sh, pj + sh]); val.append(G[new[i], new[j]])
            val = np.array(val)
            l2 = LineCollection(segs, cmap="RdBu", norm=norm, linewidths=0.6 + 2.6 * np.abs(val), capstyle="round")
            l2.set_array(0.25 - 0.75 * val); ax[1].add_collection(l2)
            for v in vac:
                ax[1].scatter(*(pos[v] + sh), marker="x", s=28, c="k", linewidths=1.3, zorder=4)
    ax[1].scatter(pos[keep, 0], pos[keep, 1], s=5, c="k", zorder=3)      # dots mark the one computed copy
    ax[1].set_title("Tiled view")
    span = 0.5 * (np.ptp(pos[:, 0]) + np.ptp(pos[:, 1])) / 2 + 0.55 * max(np.ptp(pos[:, 0]), np.ptp(pos[:, 1]))
    ax[1].set_xlim(centre[0] - span, centre[0] + span); ax[1].set_ylim(centre[1] - span, centre[1] + span)
    m = 0.5
    ax[0].set_xlim(pos[:, 0].min() - m, pos[:, 0].max() + m); ax[0].set_ylim(pos[:, 1].min() - m, pos[:, 1].max() + m)
    for x in ax:
        x.set_aspect("equal"); x.set_xticks([]); x.set_yticks([]); x.set_facecolor("0.86")
    fig.colorbar(lc, ax=ax, shrink=0.75, location="bottom", label="singlet fraction (pair entangled above 0.5)")
    fn = p[:-4] + "_map.png"; fig.savefig(fn, dpi=300, facecolor="#ffffff"); plt.close(fig); print("wrote", fn)
