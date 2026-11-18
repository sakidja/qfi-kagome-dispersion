#!/usr/bin/env python3
"""
Heat map of the QFI matrix in the wave vector basis, q against q'.

  F(q, q') = (1/N) sum_ij F_ij exp(-i q.r_i) exp(+i q'.r_j)

The diagonal F(q, q) is the QFI density f(q). Translation symmetry allows a
nonzero F(q, q') only when q - q' is a reciprocal lattice vector, so the wave
vectors are grouped by their reduced wave vector k: for the clean lattice the
matrix is block diagonal (one block per k). Vacancies break the translation
symmetry and put weight outside the blocks: the cross q coherence.

Drawn: |F(q, q')|, root mean square over the vacancy configurations (the phase of an
off block element depends on where the vacancies are, so a plain average would cancel it).
One column per number of vacancies, one row per temperature, one colour scale.

With twisted boundary conditions (results of qfi_kagome_twist.py) each twist kappa is
its own Hamiltonian and its own state. It gives a matrix among the wave vectors
Q = q + kappa of that twist only; an element between wave vectors of two different
twists does not exist. The twists therefore do not refine the q against q' grid. They
add independent samples of the same matrix, and the root mean square is taken over
configurations and twists.

With --phase the figure shows single configurations, not an average: the clean lattice
and one vacancy at two positions A and B of the same sublattice. Row 1 is |F(q, q')|.
Row 2 is the phase of F(q, q') with the origin fixed on the lattice. Row 3 is the phase
with the origin moved to the vacancy. Moving the vacancy by R multiplies every element
by exp(-i (q - q').R): the magnitude does not change, and the phase outside the blocks
records where the vacancy is. A diagonal map f(q) keeps none of this.
Extra tables with --phase: Fqq_phase_<tag>_T<T>_<column>_<row>.csv and Fqq_phase_summary_<tag>_T<T>.csv

Tables written (the raw numbers behind the figure):
  Fqq_<tag>_nv<n>_T<T>.csv    the matrix |F(q, q')| as drawn, rows and columns in the drawn order
  Fqq_order_<tag>.csv         the drawn order: position, block, qx, qy, |q|
  Fqq_share_<tag>.csv         share of sum |F|^2 outside the blocks, as in cross_q.csv

Usage: python plot_qq.py results --tag L2x3 [--Tindex 0 4] [--paper]
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
ap.add_argument("--Tindex", type=int, nargs="+", default=[0, -1], help="temperature indices to draw, one row each (default: lowest and highest)")
ap.add_argument("--gamma", type=float, default=0.5, help="colour scale exponent: 0.5 square root (default), 1 linear")
ap.add_argument("--paper", action="store_true", help="page width figure at 300 dpi")
ap.add_argument("--phase", action="store_true", help="single configurations with the phase of F(q, q'): clean, and two positions of one vacancy")
ap.add_argument("--configs", type=int, nargs=2, default=None, help="with --phase: the two one vacancy configurations to draw (default: the first pair on the same sublattice)")
ap.add_argument("--twist", type=int, nargs=2, default=[0, 0], help="with --phase on twist runs: which twist to draw (indices t1 t2)")
ap.add_argument("--floor", type=float, default=0.04, help="with --phase: the phase is drawn only where |F| exceeds this share of its largest value")
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*_t*.npz")))
twist = bool(files)                      # results of qfi_kagome_twist.py: one file per configuration and twist
if not twist:
    files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*.npz")))
if not files:
    raise SystemExit("no result files found")
runs = defaultdict(list); ntw = set()
for p in files:
    d = np.load(p)
    pos, N, qs = d["pos"], int(d["N"]), d["qs"]
    ph = np.exp(1j * qs @ pos.T)
    if twist:        # transverse channel at twist kappa: the physical wave vectors are Q = q + kappa, all inside this one twist
        runs[len(d["vac"])].append(np.einsum("qi,tij,pj->tqp", ph.conj(), d["F"], ph) / N)
        ntw.add(tuple(np.round(d["theta"], 6)))
    else:
        runs[len(d["vac"])].append(np.einsum("qi,tij,pj->tqp", ph, d["F"], ph.conj()) / N)     # (nT, nq, nq)
if twist:
    print(f"twist runs: {len(ntw)} twists; each twist gives its own {len(qs)} x {len(qs)} matrix, and the root mean square is taken over configurations and twists")
qlab, L, T, B = d["qlab"], d["L"].astype(int), d["T"], d["B"]
doses = sorted(runs); nq = len(qs)

# ---- order: group the wave vectors by reduced wave vector k, then by |q| inside each group
kred = np.column_stack([qlab[:, 0] % L[0], qlab[:, 1] % L[1]]).astype(int)
qn = np.linalg.norm(qs, axis=1)
kmin = {}
for i in range(nq):
    key = tuple(kred[i]); kmin[key] = min(kmin.get(key, 1e9), qn[i])
keys = sorted(kmin, key=lambda k: (round(kmin[k], 6), k))
block = np.array([keys.index(tuple(k)) for k in kred])
order = np.lexsort((qn, block))
blk = block[order]; edges = [0] + [i for i in range(1, nq) if blk[i] != blk[i - 1]] + [nq]
same = block[:, None] == block[None, :]

out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)
name = a.tag or "all"
lines = ["position,block,qx,qy,abs_q"]
for pnum, i in enumerate(order):
    lines.append(f"{pnum},{block[i] + 1},{qs[i, 0]:.5f},{qs[i, 1]:.5f},{qn[i]:.5f}")
open(os.path.join(out, f"Fqq_order_{name}.csv"), "w").write("\n".join(lines) + "\n")


if a.phase:
    t0 = a.Tindex[0] % len(T)
    pat = (lambda nv, c: f"{a.tag}*_nv{nv}_c{c}_t{a.twist[0]}_{a.twist[1]}.npz") if twist else (lambda nv, c: f"{a.tag}*_nv{nv}_c{c}.npz")
    def load(nv, c):
        g = sorted(glob.glob(os.path.join(a.dir, pat(nv, c))))
        return np.load(g[0]) if g else None
    d0 = load(0, 0)
    if d0 is None:
        raise SystemExit("--phase needs the clean run (zero vacancies) in the same folder: it supplies the positions of all sites")
    full = d0["pos"]                                                     # positions of all N0 sites
    def Fq(d, origin):
        pp = d["pos"] - origin; ph = np.exp(1j * d["qs"] @ pp.T); Fm = d["F"][t0]
        M_ = (np.einsum("qi,ij,pj->qp", ph.conj(), Fm, ph) if twist else np.einsum("qi,ij,pj->qp", ph, Fm, ph.conj())) / int(d["N"])
        return M_[np.ix_(order, order)]
    one = {}
    c = 0
    while True:
        d1 = load(1, c)
        if d1 is None:
            break
        one[c] = (d1, full[int(d1["vac"][0])]); c += 1
    if len(one) < 2:
        raise SystemExit("--phase needs at least two configurations with one vacancy")
    zero = np.zeros(2)
    if a.configs:
        ca, cb = a.configs
    else:                                                                # first pair related by a lattice translation: equal magnitudes
        ca = cb = None
        for i in sorted(one):
            for j in sorted(one):
                if j > i and ca is None and np.abs(np.abs(Fq(one[i][0], zero)) - np.abs(Fq(one[j][0], zero))).max() < 1e-8:
                    ca, cb = i, j
        if ca is None:
            ca, cb = sorted(one)[:2]
    cols = [("clean", d0, zero), ("position A", one[ca][0], one[ca][1]), ("position B", one[cb][0], one[cb][1])]
    FA, FB = Fq(one[ca][0], zero), Fq(one[cb][0], zero); VA, VB = Fq(*one[ca]), Fq(*one[cb])
    sameo = (blk[:, None] == blk[None, :])
    allF = np.array([Fq(one[k][0], zero) for k in sorted(one)])
    summ = ["quantity,value",
            f"temperature,{T[t0]:g}", f"configuration_A,{ca}", f"configuration_B,{cb}",
            f"vacancy_A_x,{one[ca][1][0]:.5f}", f"vacancy_A_y,{one[ca][1][1]:.5f}", f"vacancy_B_x,{one[cb][1][0]:.5f}", f"vacancy_B_y,{one[cb][1][1]:.5f}",
            f"largest_difference_of_magnitude_A_B,{np.abs(np.abs(FA) - np.abs(FB)).max():.3e}",
            f"largest_difference_of_F_A_B_fixed_origin,{np.abs(FA - FB).max():.5f}",
            f"largest_difference_of_F_A_B_origin_at_vacancy,{np.abs(VA - VB).max():.3e}",
            f"largest_imaginary_part_fixed_origin,{np.abs(FA.imag).max():.5f}",
            f"largest_imaginary_part_origin_at_vacancy,{np.abs(VA.imag).max():.3e}",
            f"elements_outside_blocks_above_floor_origin_at_vacancy,{int((np.abs(VA[~sameo]) > a.floor * np.abs(VA).max()).sum())}",
            f"of_these_negative,{int(((np.abs(VA[~sameo]) > a.floor * np.abs(VA).max()) & (VA.real[~sameo] < 0)).sum())}",
            f"sum_of_all_elements_outside_blocks_origin_at_vacancy,{VA.real[~sameo].sum():.5f}",
            f"largest_element_outside_blocks_one_configuration,{np.abs(FA[~sameo]).max():.5f}",
            f"largest_element_outside_blocks_rms_over_{len(one)}_configurations,{np.sqrt((np.abs(allF) ** 2).mean(0))[~sameo].max():.5f}",
            f"largest_element_outside_blocks_of_the_complex_average_over_{len(one)}_configurations,{np.abs(allF.mean(0))[~sameo].max():.5f}"]
    open(os.path.join(out, f"Fqq_phase_summary_{name}_T{T[t0]:g}.csv"), "w").write("\n".join(summ) + "\n"); print("\n".join(summ))
    rows = [("|F|", "mag", None), ("phase, fixed origin", "fixed", zero), ("phase, origin at vacancy", "vac", None)]
    vmax = max(np.abs(Fq(dd, zero)).max() for _, dd, _ in cols)
    plt.rcParams.update({"font.size": 8 if a.paper else 11})
    W = 5.6 if a.paper else 11.0; side = (W - 1.3) / 3
    fig, ax = plt.subplots(3, 3, figsize=(W, 3 * side * 1.05 + (0.5 if a.paper else 0.9)), squeeze=False, constrained_layout=True)
    cyc = plt.get_cmap("hsv").copy(); cyc.set_bad("#e6e6e6")
    ticks = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]
    ql = "Q" if twist else "q"
    for c, (ttl, dd, rv) in enumerate(cols):
        for r, (rl, kind, _) in enumerate(rows):
            Mx = Fq(dd, rv if kind == "vac" else zero)
            if kind == "mag":
                im0 = ax[r, c].pcolormesh(np.arange(nq + 1), np.arange(nq + 1), np.abs(Mx), cmap="viridis", norm=PowerNorm(a.gamma, vmin=0, vmax=vmax), rasterized=True)
                np.savetxt(os.path.join(out, f"Fqq_phase_{name}_T{T[t0]:g}_{ttl.replace(' ', '_')}_magnitude.csv"), np.abs(Mx), delimiter=",", fmt="%.6f")
            else:
                ang = np.ma.masked_where(np.abs(Mx) < a.floor * vmax, np.angle(Mx))
                im1 = ax[r, c].pcolormesh(np.arange(nq + 1), np.arange(nq + 1), ang, cmap=cyc, vmin=-np.pi, vmax=np.pi, rasterized=True)
                np.savetxt(os.path.join(out, f"Fqq_phase_{name}_T{T[t0]:g}_{ttl.replace(' ', '_')}_phase_{'fixed_origin' if kind == 'fixed' else 'origin_at_vacancy'}.csv"),
                           np.where(np.abs(Mx) < a.floor * vmax, np.nan, np.angle(Mx)), delimiter=",", fmt="%.6f")
            for e in edges[1:-1]:
                ax[r, c].axhline(e, color="white", lw=0.4, alpha=0.6); ax[r, c].axvline(e, color="white", lw=0.4, alpha=0.6)
            ax[r, c].set_aspect("equal"); ax[r, c].invert_yaxis(); ax[r, c].tick_params(length=0)
            ax[r, c].set_xticks(ticks); ax[r, c].set_yticks(ticks)
            ax[r, c].set_xticklabels([str(i + 1) for i in range(len(ticks))] if r == 2 else [])
            ax[r, c].set_yticklabels([str(i + 1) for i in range(len(ticks))] if c == 0 else [])
            if r == 0:
                ax[r, c].set_title(ttl if c == 0 else "vacancy at " + ttl[-1])
            if r == 2:
                ax[r, c].set_xlabel(f"{ql}′ (block)")
            if c == 0:
                ax[r, c].set_ylabel(f"{rl}\n{ql} (block)")
    fig.colorbar(im0, ax=ax[0, :], pad=0.01, aspect=12, label=f"|F({ql}, {ql}′)|")
    cb = fig.colorbar(im1, ax=ax[1:, :], pad=0.01, aspect=24, label="phase (radians)", ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    cb.ax.set_yticklabels(["−π", "−π/2", "0", "π/2", "π"])
    fn = os.path.join(out, f"Fqq_phase_{name}_T{T[t0]:g}" + ("_paper" if a.paper else "") + ".png")
    fig.savefig(fn, dpi=300 if a.paper else 140, facecolor="#ffffff"); print("wrote", fn)
    raise SystemExit(0)

tis = [t % len(T) for t in a.Tindex]
M = {}; share = ["nvac,T,share_outside_blocks,largest_element_outside_blocks,largest_diagonal_element"]
for nv in doses:
    Fa = np.array(runs[nv])                                           # (nconf, nT, nq, nq)
    for t in range(len(T)):
        rms = np.sqrt((np.abs(Fa[:, t]) ** 2).mean(axis=0))
        sh = np.mean([(np.abs(F[t][~same]) ** 2).sum() / (np.abs(F[t]) ** 2).sum() for F in Fa])
        share.append(f"{nv},{T[t]:g},{sh:.5f},{rms[~same].max() if (~same).any() else 0:.5f},{np.diag(rms).max():.5f}")
        if t in tis:
            M[nv, t] = rms[np.ix_(order, order)]
            np.savetxt(os.path.join(out, f"Fqq_{name}_nv{nv}_T{T[t]:g}.csv"), M[nv, t], delimiter=",", fmt="%.6f")
open(os.path.join(out, f"Fqq_share_{name}.csv"), "w").write("\n".join(share) + "\n")
print("\n".join(share))

# ---- figure
vmax = max(m.max() for m in M.values())
nr, nc = len(tis), len(doses)
plt.rcParams.update({"font.size": 8 if a.paper else 11})
W = min(6.5, 1.35 + 1.45 * nc) if a.paper else 1.6 + 3.2 * nc          # width follows the number of columns: no empty region
side = (W - 1.25) / nc                                                   # side of one square panel
fig, ax = plt.subplots(nr, nc, figsize=(W, nr * side * 1.06 + (0.55 if a.paper else 0.9)), squeeze=False, constrained_layout=True)
ql = "Q" if twist else "q"
ticks = [0.5 * (edges[i] + edges[i + 1]) for i in range(len(edges) - 1)]
for r, t in enumerate(tis):
    for c, nv in enumerate(doses):
        im = ax[r, c].pcolormesh(np.arange(nq + 1), np.arange(nq + 1), M[nv, t], cmap="viridis", norm=PowerNorm(a.gamma, vmin=0, vmax=vmax), rasterized=True)
        for e in edges[1:-1]:
            ax[r, c].axhline(e, color="white", lw=0.4, alpha=0.6); ax[r, c].axvline(e, color="white", lw=0.4, alpha=0.6)
        ax[r, c].set_aspect("equal"); ax[r, c].invert_yaxis()
        ax[r, c].set_xticks(ticks); ax[r, c].set_yticks(ticks)
        ax[r, c].set_xticklabels([str(i + 1) for i in range(len(ticks))] if r == nr - 1 else [])
        ax[r, c].set_yticklabels([str(i + 1) for i in range(len(ticks))] if c == 0 else [])
        ax[r, c].tick_params(length=0)
        if r == 0:
            ax[r, c].set_title("clean" if nv == 0 else f"{nv} vacanc" + ("y" if nv == 1 else "ies"))
        if r == nr - 1:
            ax[r, c].set_xlabel(f"{ql}′ (block)")
        if c == 0:
            ax[r, c].set_ylabel(f"T = {T[t]:g} J\n{ql} (block)")
fig.colorbar(im, ax=ax, pad=0.01, aspect=30, label="|F(Q, Q′)|" if twist else "|F(q, q′)|")
fn = os.path.join(out, f"Fqq_{name}" + ("_paper" if a.paper else "") + ".png")
fig.savefig(fn, dpi=300 if a.paper else 140, facecolor="#ffffff"); print("wrote", fn)
