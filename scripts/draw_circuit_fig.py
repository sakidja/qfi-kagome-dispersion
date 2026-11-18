#!/usr/bin/env python3
"""Concept figure: lattice sites as qubit wires, bonds as two qubit gates.

A schematic only, no computed data. Each Heisenberg bond J S_i.S_j equals (J/2) SWAP_ij - J/4,
so one bond is one partial SWAP gate between the two wires. Three cases, as in Section 1.1:
two spins, a triangle, and the triangle with one site removed.
Usage:  python draw_circuit_fig.py            writes circuit_concept.png (300 dpi)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--out", default="circuit_concept.png"); a = ap.parse_args()
BLUE, GREY, RED = "#1f4e79", "#b0b0b0", "#c0392b"

def lattice(ax, sites, bonds, missing=()):
    for i, j in bonds:
        dead = i in missing or j in missing
        ax.plot([sites[i][0], sites[j][0]], [sites[i][1], sites[j][1]],
                color=GREY if dead else BLUE, lw=1.5 if dead else 4, ls=":" if dead else "-", zorder=1)
    for k, (x, y) in enumerate(sites):
        if k in missing:
            ax.plot(x, y, marker="x", color=RED, ms=14, mew=3, zorder=3)
        else:
            ax.plot(x, y, marker="o", color="white", mec="black", ms=22, mew=1.5, zorder=3)
            ax.text(x, y, str(k + 1), ha="center", va="center", fontsize=11, zorder=4)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-0.75, 0.75); ax.set_ylim(-0.45, 1.15)

def circuit(ax, n, gates, missing=()):
    ys = [n - 1 - k for k in range(n)]
    L = len(gates) + 1
    for k in range(n):
        dead = k in missing
        ax.plot([0, L], [ys[k], ys[k]], color=GREY if dead else "black", lw=1 if dead else 1.5, ls=":" if dead else "-", zorder=1)
        ax.text(-0.15, ys[k], f"q{k + 1}", ha="right", va="center", fontsize=11, color=GREY if dead else "black")
    for g, (i, j) in enumerate(gates):
        dead = i in missing or j in missing
        c = GREY if dead else BLUE
        x = g + 1
        ax.plot([x, x], [ys[i], ys[j]], color=c, lw=1.5 if dead else 2.5, ls=":" if dead else "-", zorder=2)
        for k in (i, j):
            ax.plot(x, ys[k], marker="x", color=c, ms=11, mew=1.5 if dead else 3, zorder=3)
    ax.set_xlim(-0.6, L + 0.1); ax.set_ylim(-0.5, 2.5); ax.axis("off")

h = np.sqrt(3) / 2
tri = [(-0.5, 0.0), (0.5, 0.0), (0.0, h)]
cases = [("Two spins", [(-0.5, 0.35), (0.5, 0.35)], [(0, 1)], ()),
         ("Triangle", tri, [(0, 1), (1, 2), (0, 2)], ()),
         ("Triangle, one site removed", tri, [(0, 1), (1, 2), (0, 2)], (2,))]
fig, axs = plt.subplots(2, 3, figsize=(7.0, 3.6), dpi=300, gridspec_kw={"height_ratios": [1.0, 1.0]})
for c, (title, sites, bonds, missing) in enumerate(cases):
    lattice(axs[0, c], sites, bonds, missing)
    axs[0, c].set_title(title, fontsize=11)
    circuit(axs[1, c], 3 if len(sites) == 3 else 2, bonds, missing)
    if len(sites) == 2: axs[1, c].set_ylim(-1.0, 2.0)
fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.02, wspace=0.25, hspace=0.05)
fig.savefig(a.out, dpi=300)
print("wrote", a.out)
