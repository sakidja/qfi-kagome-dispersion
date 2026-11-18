#!/usr/bin/env python3
"""
Melting of the dispersion map: S(Q, w) and the QFI density at several
temperatures, side by side, for one number of vacancies.

Reads results of qfi_kagome_twist.py (all twists combined) or, if there are no
twist files, of qfi_kagome_ed.py. The runs must have been made WITH spectra.

Figure: one column per temperature.
  row 1  S(Q, w)            what a neutron experiment measures
  row 2  QFI density        the same spectrum after the tanh filter
  row 3  averaged over Q, against energy (a density of states view): the
         g density (4 S with both signs of energy folded; its area is g) and the
         f density (QFI; its area is f). Their ratio is tanh^2(energy / 2T).
The colour scale is shared along each row, so fading with temperature is seen
directly. The energy axis is cut where the weight ends (see --emax) and the
colour scale is a square root by default (see --gamma).

Tables written next to the figure (the raw numbers behind it):
  melt_profile_<tag>_nv<n>.csv   T, energy, S, g density and f density averaged over Q
  melt_weights_<tag>_nv<n>.csv   T, mean g and mean f over Q, share of the QFI weight
                                 below 0.5 J, from 0.5 to 1.5 J and above, centroid

Usage: python melt_dispersion.py tmelt_tw --tag L2x2 --nvac 1 [--T 0.02 0.3 0.7 2] [--paper]
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
ap.add_argument("--nvac", type=int, default=1)
ap.add_argument("--T", type=float, nargs="+", default=[0.02, 0.3, 0.7, 2.0], help="temperatures to draw (nearest available)")
ap.add_argument("--emax", type=float, default=None, help="top of the energy axis in J; default: where 95 percent of the weight lies below (see --weight)")
ap.add_argument("--weight", type=float, default=0.95, help="share of the spectral weight kept below the top of the energy axis")
ap.add_argument("--gamma", type=float, default=0.5, help="colour scale exponent: 0.5 square root (default), 1 linear")
ap.add_argument("--sigma", type=float, default=0.04, help="energy broadening in J; 0 shows the raw bins")
ap.add_argument("--scale", choices=["column", "shared"], default="column", help="colour scale: column = each temperature has its own (default; the bottom row keeps one common axis and shows the loss of weight), shared = one scale for all temperatures")
ap.add_argument("--paper", action="store_true", help="page width figure at 300 dpi")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv{a.nvac}_c*_t*.npz")))
twist = bool(files)
if not twist:
    files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv{a.nvac}_c*.npz")))
if not files:
    raise SystemExit("no result files found for this number of vacancies")
acc = defaultdict(list)
for p in files:
    d = np.load(p)
    if not d["Sqw"].any():
        continue
    N = int(d["N"])
    th = tuple(np.round(d["theta"], 6)) if twist else (0.0, 0.0)
    acc[th].append((d["Sqw"] / N, d["Qqw"] / N))
    qlab, B, L, T, qe = d["qlab"], d["B"], d["L"], d["T"], d["q_edges"]
if not acc:
    raise SystemExit("these runs were made without spectra (--no_spectra); run them again with spectra")
nb = len(qe) - 1; wc = 0.5 * (qe[1:] + qe[:-1]); dw = qe[1] - qe[0]

X, S, Gd, Q = [], [], [], []
for th, runs in acc.items():
    X.append(np.column_stack([(qlab[:, 0] + th[0]) / L[0], (qlab[:, 1] + th[1]) / L[1]]))
    Sm_ = np.mean([r[0] for r in runs], axis=0)                           # (nT, nq, 2 nb), already per unit energy
    S.append(Sm_[:, :, nb:])                                              # energy loss side, w > 0
    Gd.append(4 * (Sm_[:, :, nb:] + Sm_[:, :, :nb][:, :, ::-1]))          # g density: both signs of w folded, area = g(Q)
    Q.append(np.mean([r[1] for r in runs], axis=0) / dw)                  # f density: area = f(Q)
X = np.concatenate(X); S = np.concatenate(S, axis=1); Gd = np.concatenate(Gd, axis=1); Q = np.concatenate(Q, axis=1)
Qn = np.linalg.norm(X @ B, axis=1); order = np.argsort(Qn, kind="stable")
S, Gd, Q, Qn = S[:, order], Gd[:, order], Q[:, order], Qn[order]
nq = len(Qn)

# ---- tables: the raw numbers behind the figure, for every temperature
out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
name = f"{a.tag or 'all'}_nv{a.nvac}"
Sp, Gp, Qp = S.mean(axis=1), Gd.mean(axis=1), Q.mean(axis=1)              # (nT, nw)
lines = ["T,energy_J,S_mean_over_Q,g_density_mean_over_Q,f_density_mean_over_Q,tanh2_of_energy_over_2T"]
for t in range(len(T)):
    for k in range(nb):
        lines.append(f"{T[t]:g},{wc[k]:.4f},{Sp[t, k]:.6f},{Gp[t, k]:.6f},{Qp[t, k]:.6f},{np.tanh(wc[k] / (2 * T[t])) ** 2:.6f}")
open(os.path.join(out, f"melt_profile_{name}.csv"), "w").write("\n".join(lines) + "\n")
lines = ["T,mean_g_over_Q,mean_f_over_Q,share_below_0.5J,share_0.5_to_1.5J,share_above_1.5J,energy_centroid_J,f_below_0.5J,f_0.5_to_1.5J,f_above_1.5J"]
for t in range(len(T)):
    w = Qp[t] * dw; tot = w.sum()
    lines.append(f"{T[t]:g},{Gp[t].sum() * dw:.5f},{tot:.5f},{w[wc < 0.5].sum() / tot:.4f},{w[(wc >= 0.5) & (wc < 1.5)].sum() / tot:.4f},"
                 f"{w[wc >= 1.5].sum() / tot:.4f},{(w * wc).sum() / tot:.4f},"
                 f"{w[wc < 0.5].sum():.5f},{w[(wc >= 0.5) & (wc < 1.5)].sum():.5f},{w[wc >= 1.5].sum():.5f}")
open(os.path.join(out, f"melt_weights_{name}.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))

# ---- figure
ti = []
for want in a.T:
    k = int(np.argmin(np.abs(T - want)))
    if k not in ti:
        ti.append(k)
if a.emax is None:
    c = np.cumsum(Gp[ti].max(axis=0) + Qp[ti].max(axis=0)); c /= c[-1]
    emax = min(qe[-1], np.ceil(wc[np.searchsorted(c, a.weight)] / 0.25) * 0.25)
else:
    emax = a.emax
keep = wc <= emax

def smooth(M):
    if a.sigma <= 0:
        return M
    k = np.exp(-0.5 * ((wc[:, None] - wc[None, :]) / a.sigma) ** 2); k /= k.sum(axis=0, keepdims=True)
    return M @ k.T

Sm = {k: smooth(S[k])[:, keep] for k in ti}; Qm = {k: smooth(Q[k])[:, keep] for k in ti}
vS = max(m.max() for m in Sm.values()); vQ = max(m.max() for m in Qm.values())
nc = len(ti)
plt.rcParams.update({"font.size": 8 if a.paper else 12})
fig, ax = plt.subplots(3, nc, figsize=(6.5, 5.6) if a.paper else (3.6 * nc, 9), squeeze=False, constrained_layout=True)
xe = np.arange(nq + 1); ye = qe[:keep.sum() + 1]
step = max(1, nq // 4)
for c, k in enumerate(ti):
    for r, (M, v, cm) in enumerate([(Sm[k], vS, "magma"), (Qm[k], vQ, "viridis")]):
        im = ax[r, c].pcolormesh(xe, ye, M.T, cmap=cm, norm=PowerNorm(a.gamma, vmin=0, vmax=v if a.scale == "shared" else M.max()), rasterized=True)
        ax[r, c].set_xticks(xe[:-1][::step] + 0.5); ax[r, c].set_xticklabels([f"{x:.1f}" for x in Qn[::step]])
        ax[r, c].set_xlabel("|Q|")
        if c == 0:
            ax[r, c].set_ylabel("energy (J)")
        else:
            ax[r, c].set_yticklabels([])
        if a.scale == "shared":
            if c == nc - 1:
                fig.colorbar(im, ax=ax[r, :], pad=0.01, aspect=30, label=["S(Q, ω)", "QFI density"][r])
        else:
            cb = fig.colorbar(im, ax=ax[r, c], pad=0.02, aspect=25)
            cb.ax.tick_params(labelsize=6 if a.paper else 9)
            if c == nc - 1:
                cb.set_label(["S(Q, ω)", "QFI density"][r])
    ax[0, c].set_title(f"T = {T[k]:g} J")
    ax[2, c].plot(wc[keep], smooth(Gp[k][None])[0][keep], color="#c8641f", lw=1.5, label="g density")
    ax[2, c].plot(wc[keep], smooth(Qp[k][None])[0][keep], color="#1f4e79", lw=2, label="f density (QFI)")
    ax[2, c].set_xlim(0, emax); ax[2, c].set_xlabel("energy (J)")
    if c == 0:
        ax[2, c].set_ylabel("mean over Q")
    else:
        ax[2, c].set_yticklabels([])
    if c == nc - 1:
        ax[2, c].legend(frameon=False, loc="upper right")
top = max(smooth(Gp[k][None])[0][keep].max() for k in ti) * 1.05
for c in range(nc):
    ax[2, c].set_ylim(0, top)
fn = os.path.join(out, f"melt_dispersion_{name}" + ("_paper" if a.paper else "") + ".png")
fig.savefig(fn, dpi=300 if a.paper else 140, facecolor="#ffffff"); print("wrote", fn)
