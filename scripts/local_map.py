#!/usr/bin/env python3
"""
Local coherence map ("thermal heat map") from finished qfi_kagome_ed.py results.
No new runs needed.

For every bond (i, j) take the local operator O = Sz_i - Sz_j and compare
   f_b = F_ii + F_jj - 2 F_ij      its QFI (coherent part)
   g_b = G_ii + G_jj - 2 G_ij      4 x its variance (total fluctuation)
   share  s_b = f_b / g_b           1 = fully quantum, 0 = fully thermal.
For a single transition of energy w at temperature T, s = tanh^2(w / 2T), so
   coherence energy   w_b = 2 T atanh( sqrt(s_b) )
is the energy scale that protects the coherence on that bond, and
   T_eff / T = w_far / w_b
is an effective temperature relative to the bonds far from any vacancy
(w_far = their mean). T_eff < T: the bond behaves colder than the lattice.
The true temperature is uniform; T_eff is a way to read the local coherence.

When the share is saturated (above 0.995, as for an even spin count at low T)
the coherence energy is not defined and the table leaves it blank.
At low T and odd spin count, 1 - s_b shows where the unpaired spin sits.

Outputs in <dir>/analysis/: local_classes.csv (freed / near / far bonds,
averaged over configurations) and local_map_<tag>.png (one configuration per
dose; left: 1 - share at the lowest T; right: T_eff / T at the highest T).

Usage: python local_map.py results --tag L2x3 [--conf 0]
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, glob, os, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from qfi_kagome_ed import kagome_cluster

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--tag", default="")
ap.add_argument("--conf", type=int, default=0, help="configuration drawn in the map")
ap.add_argument("--paper", action="store_true", help="page width figure at 300 dpi")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*.npz")))
if not files:
    raise SystemExit("no result files found")

def analyse(p):
    d = np.load(p)
    L1, L2 = [int(x) for x in d["L"]]
    pos0, bonds0, A = kagome_cluster(L1, L2)
    vac = set(int(v) for v in d["vac"]); keep = [int(k) for k in d["keep"]]
    new = {s: k for k, s in enumerate(keep)}
    nb = defaultdict(set)
    for i, j in bonds0:
        nb[i].add(j); nb[j].add(i)
    vnb = set().union(*[nb[v] for v in vac]) - vac if vac else set()
    F, G, T = d["F"], d["G"], d["T"]
    out = dict(T=T, pos0=pos0, A=A, L=(L1, L2), vac=sorted(vac), keep=keep, bonds=[], cls=[], s=[],
               site=np.einsum("tii->ti", F) / np.einsum("tii->ti", G))
    for i, j in bonds0:
        if i in vac or j in vac:
            continue
        a_, b_ = new[i], new[j]
        f = F[:, a_, a_] + F[:, b_, b_] - 2 * F[:, a_, b_]
        g = G[:, a_, a_] + G[:, b_, b_] - 2 * G[:, a_, b_]
        out["bonds"].append((i, j)); out["s"].append(f / g)
        out["cls"].append("freed" if any(i in nb[v] and j in nb[v] for v in vac)
                          else "near" if (i in vnb or j in vnb) else "far")
    out["s"] = np.clip(np.array(out["s"]), 1e-12, 1 - 1e-12)             # (nbonds, nT)
    out["w"] = 2 * T[None, :] * np.arctanh(np.sqrt(out["s"]))
    cls = np.array(out["cls"])
    ref = out["w"][cls == "far"].mean(axis=0) if (cls == "far").any() else out["w"].mean(axis=0)
    out["teff"] = ref[None, :] / out["w"]                                 # T_eff / T
    out["cls"] = cls
    return out

res = defaultdict(dict)
for p in files:
    m = re.search(r"_nv(\d+)_c(\d+)\.npz$", p)
    res[int(m.group(1))][int(m.group(2))] = analyse(p)
doses = sorted(res); T = next(iter(res[doses[0]].values()))["T"]
out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)

lines = ["nvac,class,T,quantum_share,coherence_energy_J,Teff_over_T,Teff_sem"]
for nv in doses:
    for c in ("freed", "near", "far"):
        per = [(r["s"][r["cls"] == c].mean(0), r["w"][r["cls"] == c].mean(0), r["teff"][r["cls"] == c].mean(0))
               for r in res[nv].values() if (r["cls"] == c).any()]
        if not per:
            continue
        S, W, TE = (np.array(x) for x in zip(*per))
        for t in range(len(T)):
            sem = TE[:, t].std() / np.sqrt(len(TE)) if len(TE) > 1 else 0.0
            if S[:, t].mean() > 0.995:          # saturated: coherence energy not defined
                lines.append(f"{nv},{c},{T[t]:g},{S[:,t].mean():.4f},,,")
            else:
                lines.append(f"{nv},{c},{T[t]:g},{S[:,t].mean():.4f},{W[:,t].mean():.4f},{TE[:,t].mean():.4f},{sem:.4f}")
open(os.path.join(out, "local_classes.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

tsel = [0, len(T) - 1]
if a.paper:
    plt.rcParams.update({"font.size": 9})
LW, MS, XS = (4, 22, 70) if a.paper else (7, 60, 180)
fig, ax = plt.subplots(len(doses), 2, figsize=((6.5, 2.5 * len(doses) + 0.8) if a.paper else (11, 4.2 * len(doses))),
                       squeeze=False, constrained_layout=True)
norm = LogNorm(vmin=0.6, vmax=1.67)
for r_, nv in enumerate(doses):
    R = res[nv].get(a.conf, next(iter(res[nv].values())))
    pos0, A, (L1, L2) = R["pos0"], R["A"], R["L"]
    for c_, t in enumerate(tsel):
        x = ax[r_, c_]
        val = (1.0 - R["s"][:, t]) if c_ == 0 else R["teff"][:, t]
        segs, cols = [], []
        for k, (i, j) in enumerate(R["bonds"]):
            dv = None
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    v = pos0[j] - pos0[i] + n1 * L1 * A[0] + n2 * L2 * A[1]
                    if abs(np.linalg.norm(v) - 1) < 1e-9:
                        dv = v
            if np.allclose(pos0[i] + dv, pos0[j]):
                segs.append([pos0[i], pos0[j]]); cols.append(val[k])
            else:                                   # bond across the periodic edge: two half bonds
                segs.append([pos0[i], pos0[i] + dv / 2]); cols.append(val[k])
                segs.append([pos0[j], pos0[j] - dv / 2]); cols.append(val[k])
        if c_ == 0:
            lc0 = LineCollection(segs, array=np.array(cols), cmap="Reds", linewidths=LW, capstyle="round")
            lc0.set_clim(0, 0.4); x.add_collection(lc0)
        else:
            lc1 = LineCollection(segs, array=np.array(cols), cmap="coolwarm", norm=norm, linewidths=LW,
                                 capstyle="round")
            x.add_collection(lc1)
        x.scatter(pos0[R["keep"], 0], pos0[R["keep"], 1], c="0.25", s=MS, zorder=3)
        if R["vac"]:
            x.scatter(pos0[R["vac"], 0], pos0[R["vac"], 1], marker="x", c="k", s=XS, linewidths=2 if a.paper else 3.5, zorder=4)
        x.set_aspect("equal"); x.set_xticks([]); x.set_yticks([]); x.margins(0.12)
        if r_ == 0:
            x.set_title(f"T = {T[t]:g} J: share that is not quantum" if c_ == 0
                        else f"T = {T[t]:g} J: effective temperature")
        if c_ == 0:
            x.set_ylabel(f"{nv} vacancies")
cb0 = fig.colorbar(lc0, ax=ax[:, 0], shrink=0.6, location="bottom")
cb0.set_label("1 - quantum share (red: unpaired spin sits here)")
cb1 = fig.colorbar(lc1, ax=ax[:, 1], shrink=0.6, location="bottom", ticks=[0.6, 0.8, 1, 1.25, 1.67])
cb1.ax.set_xticklabels(["0.6", "0.8", "1", "1.25", "1.67"]); cb1.set_label("bond T_eff / T (blue: colder, red: hotter)")
if not a.paper:
    fig.suptitle("Local coherence map (x = vacancy)", fontsize=14)
fn = os.path.join(out, f"local_map_{a.tag or 'all'}" + ("_paper" if a.paper else "") + ".png")
fig.savefig(fn, dpi=300 if a.paper else 130, facecolor="#ffffff"); print("wrote", fn)
