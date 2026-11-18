#!/usr/bin/env python3
"""
Magnetic dispersion from qfi_kagome_twist.py results: all twists combined, so
the wave vector axis is nt1 x nt2 times finer than the cluster alone.

Each file holds one twist; its physical wave vectors are Q = q + kappa.
Configurations of the same dose and twist are averaged.

--line b1 | b2 : points on the line through the origin along b1 or b2
--line all     : every Q point, ordered by |Q| (powder view)
Left: S(Q, w). Right: energy resolved QFI. Also writes fQ_<...>.csv with
f(Q) and g(Q) for every Q point, dose and temperature.

Usage: python plot_twist.py results_twist --tag L2x3 --line b2
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
ap.add_argument("--sigma", type=float, default=0.04)
ap.add_argument("--scale", choices=["row", "shared"], default="row", help="colour scale: row = each row has its own (default, read each colour bar), shared = one scale for all rows")
ap.add_argument("--line", choices=["all", "b1", "b2"], default="b2")
ap.add_argument("--paper", action="store_true",
                help="page width figure at 300 dpi: one dose is drawn as three stacked panels, several doses as a grid")
ap.add_argument("--nvac", type=int, default=None, help="draw only this number of vacancies")
a = ap.parse_args()

acc = defaultdict(lambda: defaultdict(list))          # acc[nv][(t1,t2)] -> list of dicts
for p in sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*_t*.npz"))):
    d = np.load(p)
    N = int(d["N"])
    acc[len(d["vac"])][tuple(np.round(d["theta"], 6))].append(
        dict(f=d["f"], g=d["g"], S=d["Sqw"][a.Tindex] / N, Q=d["Qqw"][a.Tindex] / N))
    qlab, B, L, T, qe, kap_of = d["qlab"], d["B"], d["L"], d["T"], d["q_edges"], None
if not acc:
    raise SystemExit("no twist files found")
doses = sorted(acc)
nb = len(qe) - 1; wc = 0.5 * (qe[1:] + qe[:-1]); dw = qe[1] - qe[0]
has_spec = any(r["S"].any() for nv in acc for th in acc[nv] for r in acc[nv][th])

def table(nv):
    """Combine twists: fractional coordinates x1, x2 with Q = x1 b1 + x2 b2."""
    X, F, G, S, Qd = [], [], [], [], []
    for th, runs in acc[nv].items():
        X.append(np.column_stack([(qlab[:, 0] + th[0]) / L[0], (qlab[:, 1] + th[1]) / L[1]]))
        F.append(np.mean([r["f"] for r in runs], axis=0).T)      # (nq, nT)
        G.append(np.mean([r["g"] for r in runs], axis=0).T)
        S.append(np.mean([r["S"] for r in runs], axis=0)[:, nb:])
        Qd.append(np.mean([r["Q"] for r in runs], axis=0) / dw)
    return [np.concatenate(v) for v in (X, F, G, S, Qd)]

def smooth(M):
    if a.sigma <= 0:
        return M
    k = np.exp(-0.5 * ((wc[:, None] - wc[None, :]) / a.sigma) ** 2); k /= k.sum(axis=0, keepdims=True)
    return M @ k.T

out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
data = {nv: table(nv) for nv in doses}
lines = ["nvac,T,x1,x2,Qx,Qy,absQ,f,g"]
for nv in doses:
    X, F, G, _, _ = data[nv]
    Qc = X @ B
    for k in np.lexsort((X[:, 1], X[:, 0])):
        for t in range(len(T)):
            lines.append(f"{nv},{T[t]:g},{X[k,0]:.5f},{X[k,1]:.5f},{Qc[k,0]:.5f},{Qc[k,1]:.5f},"
                         f"{np.linalg.norm(Qc[k]):.5f},{F[k,t]:.6f},{G[k,t]:.6f}")
open(os.path.join(out, f"fQ_{a.tag or 'all'}.csv"), "w").write("\n".join(lines) + "\n")
print(f"{len(data[doses[0]][0])} Q points per dose; wrote fQ csv")

def pick(X):
    Qn = np.linalg.norm(X @ B, axis=1)
    if a.line == "b1":
        s = np.where(np.abs(X[:, 1]) < 1e-9)[0]; s = s[np.argsort(X[s, 0])]; xv = X[s, 0]
    elif a.line == "b2":
        s = np.where(np.abs(X[:, 0]) < 1e-9)[0]; s = s[np.argsort(X[s, 1])]; xv = X[s, 1]
    else:
        s = np.argsort(Qn, kind="stable"); xv = Qn[s]
    return s, xv

if a.nvac is not None:
    doses = [nv for nv in doses if nv == a.nvac]
ncol = 3 if has_spec else 1
STACK = a.paper and len(doses) == 1
plt.rcParams.update({"font.size": (10 if STACK else 7) if a.paper else 13})
if STACK:
    fig, ax = plt.subplots(ncol, 1, figsize=(6.5, 2.8 * ncol), constrained_layout=True); ax = np.array([ax]).reshape(1, ncol)
elif a.paper:
    fig, ax = plt.subplots(len(doses), ncol, figsize=(6.5, 1.9 * len(doses) + 0.4), squeeze=False, constrained_layout=True)
else:
    fig, ax = plt.subplots(len(doses), ncol, figsize=(6.2 * ncol, 3.2 * len(doses)), squeeze=False,
                           constrained_layout=True)
xlabel = {"b1": "Q along b1 (units of b1)", "b2": "Q along b2 (units of b2)", "all": "|Q|"}[a.line]
if a.emax is None:        # fit the energy axis to the data: no empty upper part
    tot = sum(data[nv][3].sum(axis=0) + data[nv][4].sum(axis=0) for nv in doses); cum = np.cumsum(tot) / tot.sum()
    a.emax = float(min(qe[-1], np.ceil(wc[np.searchsorted(cum, a.weight)] / 0.25) * 0.25)) if has_spec else 3.0
keep = wc <= a.emax
prep = {}
for nv in doses:
    X, F, G, S, Qd = data[nv]
    s, xv = pick(X)
    prep[nv] = (xv, F[s, a.Tindex], G[s, a.Tindex], smooth(S[s])[:, keep], smooth(Qd[s])[:, keep])
vS = max(p[3].max() for p in prep.values()) if has_spec else 1
vQ = max(p[4].max() for p in prep.values()) if has_spec else 1
for r, nv in enumerate(doses):
    xv, f, g, S, Qd = prep[nv]
    ax[r, 0].plot(xv, g, color="#c8641f", lw=1.5, marker="o", ms=3, label="g (equal time)")
    ax[r, 0].plot(xv, f, color="#1f4e79", lw=2, marker="o", ms=3, label="f (QFI)")
    ax[r, 0].set_ylabel(f"{nv} vacancies"); ax[r, 0].set_ylim(0, None)
    if r == 0:
        ax[r, 0].set_title("f(Q) and g(Q)"); ax[r, 0].legend(frameon=False, fontsize=6 if (a.paper and not STACK) else 10)
    if has_spec:
        xe = np.arange(len(xv) + 1)
        for c, (M, v, cm, name) in enumerate([(S, vS if a.scale == "shared" else S.max(), "magma", "S(Q, ω)"), (Qd, vQ if a.scale == "shared" else Qd.max(), "viridis", "QFI density")]):
            im = ax[r, c + 1].pcolormesh(xe, qe[:keep.sum() + 1], M.T, cmap=cm, norm=PowerNorm(a.gamma, vmin=0, vmax=v), rasterized=True)
            step = max(1, len(xv) // (4 if (a.paper and not STACK) else 8))   # fewer labels on the narrow panels
            ax[r, c + 1].set_xticks(xe[:-1][::step] + 0.5)
            ax[r, c + 1].set_xticklabels([f"{x:.1f}" if (a.paper and not STACK) else f"{x:.2f}" for x in xv[::step]], fontsize=7 if (a.paper and not STACK) else 9)
            ax[r, c + 1].set_ylabel("energy (J)")
            if r == 0:
                ax[r, c + 1].set_title(name)
            fig.colorbar(im, ax=ax[r, c + 1], pad=0.01)
    for c in range(ncol):
        if r == len(doses) - 1 and (not STACK or c == ncol - 1):
            ax[r, c].set_xlabel(xlabel)
if not a.paper:
    fig.suptitle(f"Twist combined magnetic dispersion, T = {T[a.Tindex]:g} J", fontsize=14)
fn = os.path.join(out, f"twist_dispersion_T{T[a.Tindex]:g}_{a.line}" + ("_paper" if a.paper else "") + ".png")
fig.savefig(fn, dpi=300 if a.paper else 140, facecolor="#ffffff"); print("wrote", fn)
