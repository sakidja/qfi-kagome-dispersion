#!/usr/bin/env python3
"""
Cold finger chain: bond correlation by distance from the hole. No new runs.

Each surviving site gets d = number of bonds to the nearest vacancy (on the
full lattice). Each bond is labelled by the pair (d_i, d_j):
   (1,1)  the freed pair: both ends are neighbours of the same vacancy
   (1,2)  from a locked node into its other triangle
   (2,2)  the opposite bond of that triangle
   (2,3), (3,3), ...  farther out
For every label it reports the mean bond correlation G = 4 <Sz_i Sz_j>_c
(perfect singlet -1), the bond's quantum share s = f/g for O = Sz_i - Sz_j,
and how many bonds of that kind a configuration has.
Open system reading: the thermal state is spin rotation invariant, so the
reduced state of a bond is fixed by its singlet fraction p_s = 1/4 - 3G/4
(1 = pure singlet, decoupled from the rest; 1/4 = fully mixed). p_s > 1/2
means the two spins are entangled with each other as a pair.

Usage: python shell_test.py results --tag L2x3 [--nvac 1]
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, glob, os
from collections import defaultdict, deque
import numpy as np
from qfi_kagome_ed import kagome_cluster

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--tag", default="")
ap.add_argument("--nvac", type=int, default=1, help="dose to analyse (1 is the cleanest reading)")
a = ap.parse_args()
acc = defaultdict(lambda: dict(G=[], s=[], n=[]))
T = None
for p in sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv{a.nvac}_c*.npz"))):
    d = np.load(p); T = d["T"]
    L1, L2 = [int(x) for x in d["L"]]
    pos0, bonds0, _ = kagome_cluster(L1, L2)
    vac = [int(v) for v in d["vac"]]; keep = [int(k) for k in d["keep"]]; new = {s: k for k, s in enumerate(keep)}
    nb = defaultdict(set)
    for i, j in bonds0:
        nb[i].add(j); nb[j].add(i)
    dist = {v: 0 for v in vac}; q = deque(vac)
    while q:
        u = q.popleft()
        for w in nb[u]:
            if w not in dist:
                dist[w] = dist[u] + 1; q.append(w)
    F, G = d["F"], d["G"]
    per = defaultdict(lambda: ([], []))
    for i, j in bonds0:
        if i in vac or j in vac:
            continue
        lab = tuple(sorted((dist[i], dist[j])))
        if lab == (1, 1) and not any(i in nb[v] and j in nb[v] for v in vac):
            lab = (1, 1, "different holes")
        x, y = new[i], new[j]
        f = F[:, x, x] + F[:, y, y] - 2 * F[:, x, y]; g = G[:, x, x] + G[:, y, y] - 2 * G[:, x, y]
        per[lab][0].append(G[:, x, y]); per[lab][1].append(f / g)
    for lab, (gg, ss) in per.items():
        acc[lab]["G"].append(np.mean(gg, axis=0)); acc[lab]["s"].append(np.mean(ss, axis=0)); acc[lab]["n"].append(len(gg))
if not acc:
    raise SystemExit("no result files for that dose")
out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
lines = ["label,bonds_per_config,T,G_bond,G_sem,quantum_share,share_sem,singlet_fraction,pair_entangled"]
for lab in sorted(acc, key=lambda l: (sum(l[:2]), len(l))):
    Gm, Sm = np.array(acc[lab]["G"]), np.array(acc[lab]["s"]); n = len(Gm)
    for t in range(len(T)):
        sem = lambda x: x[:, t].std() / np.sqrt(n) if n > 1 else 0.0
        name = "(" + ",".join(str(v) for v in lab[:2]) + ")" + (" diff holes" if len(lab) > 2 else "")
        lines.append(f"\"{name}\",{np.mean(acc[lab]['n']):.1f},{T[t]:g},{Gm[:,t].mean():.4f},{sem(Gm):.4f},{Sm[:,t].mean():.4f},{sem(Sm):.4f},{0.25-0.75*Gm[:,t].mean():.4f},{'yes' if 0.25-0.75*Gm[:,t].mean()>0.5 else 'no'}")
open(os.path.join(out, f"shells_nv{a.nvac}.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
