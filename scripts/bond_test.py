#!/usr/bin/env python3
"""
Singlet test on finished qfi_kagome_ed.py results. No new runs needed.

For every nearest neighbour bond (i, j) it reads G_ij = 4 <S_i^z S_j^z>_connected
from the stored file. Reference values:
    perfect singlet        G = -1
    uncorrelated spins     G =  0
    clean kagome average   G = 4 E0 / (3 J N_bonds)   (pass --E0clean, --nbonds)

Bonds are sorted by where they sit relative to the vacancies:
    freed  : both ends were neighbours of the same vacancy (the bond left
             over from a triangle that lost one corner)
    near   : touches a neighbour of a vacancy, but is not a freed bond
    far    : everything else
If vacancies pin singlets, the freed bonds sit closer to -1 than the far ones.

Usage: python bond_test.py results --tag L2x3 --E0clean -8.0482707735 --nbonds 36
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, glob, os
from collections import defaultdict
import numpy as np
from qfi_kagome_ed import kagome_cluster

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--tag", default="")
ap.add_argument("--E0clean", type=float, default=None, help="clean ground energy from the nv0 log")
ap.add_argument("--nbonds", type=int, default=None, help="bond count of the clean cluster")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*.npz")))
if not files:
    raise SystemExit("no result files found")
acc = defaultdict(lambda: defaultdict(list))      # acc[nvac][class] -> list of (nT,) arrays
T = None
for p in files:
    d = np.load(p)
    T = d["T"]
    L1, L2 = [int(x) for x in d["L"]]
    _, bonds0, _ = kagome_cluster(L1, L2)
    vac = set(int(v) for v in d["vac"])
    keep = [int(k) for k in d["keep"]]
    new = {s: k for k, s in enumerate(keep)}
    nb = defaultdict(set)
    for i, j in bonds0:
        nb[i].add(j); nb[j].add(i)
    vnb = set().union(*[nb[v] for v in vac]) - vac if vac else set()
    G = d["G"]
    cls = defaultdict(list)
    for i, j in bonds0:
        if i in vac or j in vac:
            continue
        g = G[:, new[i], new[j]]
        if any(i in nb[v] and j in nb[v] for v in vac):
            cls["freed"].append(g)
        elif i in vnb or j in vnb:
            cls["near"].append(g)
        else:
            cls["far"].append(g)
    for c, v in cls.items():
        acc[len(vac)][c].append(np.mean(v, axis=0))

out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
lines = ["nvac,class,nconf,T,G_bond_mean,G_bond_sem"]
for nv in sorted(acc):
    for c in ("freed", "near", "far"):
        if c not in acc[nv]:
            continue
        arr = np.array(acc[nv][c])
        for t, TT in enumerate(T):
            sem = arr[:, t].std() / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            lines.append(f"{nv},{c},{len(arr)},{TT:g},{arr[:, t].mean():.5f},{sem:.5f}")
open(os.path.join(out, "bond_test.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("\nreference: perfect singlet G = -1.00000")
if a.E0clean is not None and a.nbonds:
    print(f"reference: clean kagome ground state, bond average G = {4*a.E0clean/(3*a.nbonds):.5f}")
