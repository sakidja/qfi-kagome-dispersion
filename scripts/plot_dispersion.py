#!/usr/bin/env python3
"""
Magnetic dispersion from finished qfi_kagome_ed.py results (runs made with
spectra). The spin analog of a phonon dispersion plot: energy transfer on the
vertical axis, wave vector on the horizontal axis, intensity as colour.

Left column : S(q, w), what inelastic neutron scattering measures.
Right column: the energy resolved QFI, the same spectrum after the tanh
              filter, i.e. the coherent quantum part only.
One row per vacancy dose, averaged over configurations.

The q points are the commensurate points of the cluster, ordered by |q|
(the view a powder experiment gives). Use --line b1 or --line b2 to show
only the points on one reciprocal lattice direction instead.
--sigma broadens in energy (in J) for readability; --sigma 0 shows raw bins.

Usage: python plot_dispersion.py results --tag L2x3 [--Tindex 0] [--emax 3]
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--tag", default="")
ap.add_argument("--Tindex", type=int, default=0)
ap.add_argument("--emax", type=float, default=None, help="top of the energy axis in J; default: where 95 percent of the weight lies below (see --weight)")
ap.add_argument("--weight", type=float, default=0.95, help="share of the spectral weight kept below the top of the energy axis")
ap.add_argument("--gamma", type=float, default=0.5, help="colour scale exponent: 0.5 square root (default), 1 linear")
ap.add_argument("--sigma", type=float, default=0.05)
ap.add_argument("--line", choices=["all", "b1", "b2"], default="all")
a = ap.parse_args()

S, Q = defaultdict(list), defaultdict(list)
for p in sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*.npz"))):
    d = np.load(p)
    if not d["Sqw"].any():
        continue
    N = int(d["N"]); nv = len(d["vac"])
    S[nv].append(d["Sqw"][a.Tindex] / N); Q[nv].append(d["Qqw"][a.Tindex] / N)
    qs, qlab, T, se, qe = d["qs"], d["qlab"], d["T"], d["s_edges"], d["q_edges"]
if not S:
    raise SystemExit("no files with spectra found")
doses = sorted(S)
nb = len(qe) - 1
wc = 0.5 * (qe[1:] + qe[:-1]); dw = qe[1] - qe[0]

if a.line == "b1":
    sel = np.where(qlab[:, 1] == 0)[0]; sel = sel[np.argsort(qlab[sel, 0])]
elif a.line == "b2":
    sel = np.where(qlab[:, 0] == 0)[0]; sel = sel[np.argsort(qlab[sel, 1])]
else:
    sel = np.argsort(np.linalg.norm(qs, axis=1), kind="stable")
qn = np.linalg.norm(qs[sel], axis=1)

def smooth(M):
    if a.sigma <= 0:
        return M
    k = np.exp(-0.5 * ((wc[:, None] - wc[None, :]) / a.sigma) ** 2)
    k /= k.sum(axis=0, keepdims=True)
    return M @ k.T

if a.emax is None:        # fit the energy axis to the data: no empty upper part
    tot = sum(np.mean(S[nv], axis=0)[:, nb:].sum(axis=0) + np.mean(Q[nv], axis=0).sum(axis=0) / dw for nv in doses)
    cum = np.cumsum(tot) / tot.sum(); a.emax = float(min(qe[-1], np.ceil(wc[np.searchsorted(cum, a.weight)] / 0.25) * 0.25))
keep = wc <= a.emax
plt.rcParams.update({"font.size": 13})
fig, ax = plt.subplots(len(doses), 2, figsize=(13, 3.3 * len(doses)), squeeze=False,
                       constrained_layout=True)
Sm = {nv: smooth(np.mean(S[nv], axis=0)[sel][:, nb:]) for nv in doses}        # w > 0 side
Qm = {nv: smooth(np.mean(Q[nv], axis=0)[sel] / dw) for nv in doses}
vS = max(m[:, keep].max() for m in Sm.values()); vQ = max(m[:, keep].max() for m in Qm.values())
x = np.arange(len(sel) + 1)
for r, nv in enumerate(doses):
    for c, (M, v, cm, name) in enumerate([(Sm[nv], vS, "magma", "S(q, ω)"),
                                         (Qm[nv], vQ, "viridis", "QFI density")]):
        im = ax[r, c].pcolormesh(x, qe[:keep.sum() + 1], M[:, keep].T, cmap=cm, norm=PowerNorm(a.gamma, vmin=0, vmax=v),
                                 shading="flat", rasterized=True)
        ax[r, c].set_ylabel(f"{nv} vacancies\nenergy (J)" if c == 0 else "energy (J)")
        ax[r, c].set_xticks(x[:-1] + 0.5)
        ax[r, c].set_xticklabels([f"{v_:.1f}" for v_ in qn], rotation=90, fontsize=8)
        if r == 0:
            ax[r, c].set_title(name)
        if r == len(doses) - 1:
            ax[r, c].set_xlabel("|q| of each commensurate point" if a.line == "all"
                                else f"|q| along {a.line}")
        fig.colorbar(im, ax=ax[r, c], pad=0.01)
fig.suptitle(f"Magnetic dispersion, T = {T[a.Tindex]:g} J", fontsize=14)
out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
fn = os.path.join(out, f"dispersion_T{T[a.Tindex]:g}_{a.line}.png")
fig.savefig(fn, dpi=140, facecolor="#ffffff"); print("wrote", fn)
