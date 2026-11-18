#!/usr/bin/env python3
"""
Melting of the bond pattern around a vacancy, as a picture sequence and a GIF.
No new runs: it reads finished qfi_kagome_ed.py results.

One frame per temperature in the files. Each bond of one configuration is
drawn with
   colour = singlet fraction p_s = 1/4 - 3G/4  (blue above 1/2: the two spins
            are entangled as a pair; red below 1/2; white at 1/2)
   width  = |G|, the bond correlation (1 for a perfect singlet).
The cluster is drawn centred on the vacancy (cross). Bonds that cross the
periodic edge are drawn as two half bonds.

Outputs in <dir>/analysis/:
   melt_<tag>_nv<n>_c<c>_frame_XX.png   one picture per temperature
   melt_<tag>_nv<n>_c<c>_strip.png      all temperatures side by side
   melt_<tag>_nv<n>_c<c>.gif            the animation

Usage: python melt_movie.py results --tag L2x3 [--nvac 1] [--conf 0] [--ms 900]
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
from qfi_kagome_ed import kagome_cluster

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--tag", default="L2x3")
ap.add_argument("--nvac", type=int, default=1); ap.add_argument("--conf", type=int, default=0)
ap.add_argument("--ms", type=int, default=900, help="time per frame in the GIF")
a = ap.parse_args()
p = os.path.join(a.dir, f"{a.tag}_nv{a.nvac}_c{a.conf}.npz")
d = np.load(p)
T, G = d["T"], d["G"]
L1, L2 = [int(x) for x in d["L"]]
pos0, bonds0, A = kagome_cluster(L1, L2)
vac = [int(v) for v in d["vac"]]; keep = [int(k) for k in d["keep"]]; new = {s: k for k, s in enumerate(keep)}
T1, T2 = L1 * A[0], L2 * A[1]
centre = pos0[vac[0]] if vac else pos0.mean(axis=0)

def nearest(r, ref):
    """periodic image of r closest to ref"""
    best = min(((n1, n2) for n1 in (-2, -1, 0, 1, 2) for n2 in (-2, -1, 0, 1, 2)),
               key=lambda n: np.linalg.norm(r + n[0] * T1 + n[1] * T2 - ref))
    return r + best[0] * T1 + best[1] * T2

pos = np.array([nearest(r, centre) for r in pos0])
segs, idx = [], []
for i, j in bonds0:
    if i not in new or j not in new:
        continue
    pj = nearest(pos0[j], pos[i])                 # image of j that is bonded to i
    if np.allclose(pj, pos[j]):
        segs.append([pos[i], pos[j]]); idx.append((new[i], new[j]))
    else:                                         # across the edge: two half bonds
        pi = nearest(pos0[i], pos[j])
        segs.append([pos[i], 0.5 * (pos[i] + pj)]); idx.append((new[i], new[j]))
        segs.append([pos[j], 0.5 * (pos[j] + pi)]); idx.append((new[i], new[j]))
ii = np.array([k[0] for k in idx]); jj = np.array([k[1] for k in idx])
norm = TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=1.0)
out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
stem = os.path.join(out, f"melt_{a.tag}_nv{a.nvac}_c{a.conf}")

def draw(ax, t):
    g = G[t][ii, jj]; ps = 0.25 - 0.75 * g
    lc = LineCollection(segs, cmap="RdBu", norm=norm, linewidths=1.5 + 9.0 * np.abs(g), capstyle="round")
    lc.set_array(ps); ax.add_collection(lc)
    ax.scatter(pos[keep, 0], pos[keep, 1], s=28, c="k", zorder=3)
    for v in vac:
        ax.scatter(*pos[v], marker="x", s=160, c="k", linewidths=3, zorder=4)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    m = 0.6
    ax.set_xlim(pos[:, 0].min() - m, pos[:, 0].max() + m); ax.set_ylim(pos[:, 1].min() - m, pos[:, 1].max() + m)
    ax.set_title(f"T = {T[t]:g} J", fontsize=13)
    return lc

frames = []
for t in range(len(T)):
    fig, ax = plt.subplots(figsize=(5.4, 5.2), constrained_layout=True)
    lc = draw(ax, t)
    fig.colorbar(lc, ax=ax, shrink=0.8, label="singlet fraction (pair entangled above 0.5)")
    f = f"{stem}_frame_{t:02d}.png"; fig.savefig(f, dpi=130); plt.close(fig); frames.append(f)

n = len(T); cols = min(n, 5); rows = (n + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.7 * rows), squeeze=False, constrained_layout=True)
for k, ax in enumerate(axs.ravel()):
    if k < n:
        lc = draw(ax, k)
    else:
        ax.axis("off")
fig.colorbar(lc, ax=axs, shrink=0.7, label="singlet fraction (pair entangled above 0.5)")
fig.savefig(stem + "_strip.png", dpi=150); plt.close(fig)
print("wrote", stem + "_strip.png", "and", len(frames), "frames")
try:
    from PIL import Image
    ims = [Image.open(f).convert("P", palette=Image.ADAPTIVE) for f in frames]
    ims[0].save(stem + ".gif", save_all=True, append_images=ims[1:], duration=a.ms, loop=0)
    print("wrote", stem + ".gif")
except Exception as e:
    print("GIF not written (", e, "); the frame files are complete")
