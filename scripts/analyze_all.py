#!/usr/bin/env python3
"""
Further analysis of finished qfi_kagome_ed.py results. No new runs needed.
Usage: python analyze_all.py results --tag L2x3

Tables written to <dir>/analysis/ and printed:
 A  same_q.csv     f and g at one fixed wave vector q* for every dose and T
                   (q* = brightest f point of the lowest dose at the lowest T),
                   so coherence and correlation are compared at the same q.
 B  zones.csv      mean f over first Brillouin zone points (f1, in phase) and
                   over extended zone points (f2, out of phase), and f2/f1.
                   Constant ratio with dose = even (congruent) change;
                   changing ratio = channel selective (incongruent).
 C  cross_q.csv    share of the QFI matrix weight that links different wave
                   vectors, sum |F(q,q')|^2 over q - q' not a reciprocal
                   lattice vector, divided by the total. Exactly 0 for a
                   clean lattice; vacancies switch it on.
 D  witness_T.csv  temperature where the peak f falls through 1 (pairs no
                   longer witnessed), by linear interpolation in T.
 F  averaged.csv   peak and spread (H_norm) of the configuration AVERAGED map,
                   next to the mean of the per configuration values. A crystal
                   measures the average over defect positions, and taking the
                   maximum (or the entropy) configuration by configuration is
                   biased once each configuration breaks the lattice symmetry.
 E  energy.csv     where the QFI at q* sits in energy: centroid, and the
                   share below 0.5 J (needs runs made with spectra).
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, glob, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--tag", default="")
a = ap.parse_args()
files = sorted(glob.glob(os.path.join(a.dir, f"{a.tag}*_nv*_c*.npz")))
if not files:
    raise SystemExit("no result files found")

runs = defaultdict(list)
for p in files:
    d = np.load(p)
    pos, N, qs, T = d["pos"], int(d["N"]), d["qs"], d["T"]
    ph = np.exp(1j * qs @ pos.T)                                   # (nq, N)
    Fqq = np.einsum("qi,tij,pj->tqp", ph, d["F"], ph.conj()) / N   # (nT, nq, nq)
    Gqq = np.einsum("qi,tij,qj->tq", ph, d["G"], ph.conj()).real / N
    runs[len(d["vac"])].append(dict(f=np.einsum("tqq->tq", Fqq).real, g=Gqq, Fqq=Fqq,
                                    Q=d["Qqw"] / N, N=N))
qlab, B, L = d["qlab"], d["B"], d["L"]
q_edges = d["q_edges"]
doses = sorted(runs); nT, nq = len(T), len(qs)
mean = lambda nv, k: np.mean([r[k] for r in runs[nv]], axis=0)
sem = lambda x: (np.std(x, axis=0) / np.sqrt(len(x))) if len(x) > 1 else np.zeros_like(x[0])
out = os.path.join(a.dir, "analysis"); os.makedirs(out, exist_ok=True)

def write(name, lines):
    open(os.path.join(out, name), "w").write("\n".join(lines) + "\n")
    print(f"\n== {name}"); print("\n".join(lines))

# A: same q
iq = int(np.argmax(mean(doses[0], "f")[0]))
lines = [f"# q* = ({qs[iq,0]:.5f}, {qs[iq,1]:.5f})", "nvac,T,f_at_qstar,f_sem,g_at_qstar,g_sem,f_over_g"]
for nv in doses:
    fa = np.array([r["f"][:, iq] for r in runs[nv]]); ga = np.array([r["g"][:, iq] for r in runs[nv]])
    for t in range(nT):
        lines.append(f"{nv},{T[t]:g},{fa[:,t].mean():.5f},{sem(fa)[t]:.5f},{ga[:,t].mean():.5f},"
                     f"{sem(ga)[t]:.5f},{fa[:,t].mean()/ga[:,t].mean():.5f}")
write("same_q.csv", lines)

# B: zones. First zone = closer to the origin than to any other reciprocal lattice point.
Gs = np.array([i * B[0] + j * B[1] for i in (-1, 0, 1) for j in (-1, 0, 1) if (i, j) != (0, 0)])
dist = np.linalg.norm(qs[:, None, :] - Gs[None, :, :], axis=2).min(axis=1)
first = np.linalg.norm(qs, axis=1) <= dist + 1e-9
lines = [f"# {first.sum()} first zone points (zone boundary included), {(~first).sum()} extended zone points",
         "nvac,T,f1_first_zone,f2_extended_zone,ratio_f2_over_f1,ratio_sem"]
for nv in doses:
    f1 = np.array([r["f"][:, first].mean(axis=1) for r in runs[nv]])
    f2 = np.array([r["f"][:, ~first].mean(axis=1) for r in runs[nv]])
    for t in range(nT):
        lines.append(f"{nv},{T[t]:g},{f1[:,t].mean():.5f},{f2[:,t].mean():.5f},"
                     f"{(f2[:,t]/f1[:,t]).mean():.5f},{sem(f2/f1)[t]:.5f}")
write("zones.csv", lines)

# C: cross q coherence
dn = qlab[:, None, :] - qlab[None, :, :]
same = (dn[..., 0] % int(L[0]) == 0) & (dn[..., 1] % int(L[1]) == 0)
lines = ["nvac,T,cross_q_share,cross_q_sem"]
for nv in doses:
    sh = np.array([[(np.abs(r["Fqq"][t][~same]) ** 2).sum() / (np.abs(r["Fqq"][t]) ** 2).sum()
                    for t in range(nT)] for r in runs[nv]])
    for t in range(nT):
        lines.append(f"{nv},{T[t]:g},{sh[:,t].mean():.5f},{sem(sh)[t]:.5f}")
write("cross_q.csv", lines)

# D: temperature where peak f falls through 1, for the averaged map and for the mean of the per configuration peaks
def cross1(fm):
    if fm.min() > 1: return "above 1 at all T"
    if fm.max() < 1: return "below 1 at all T"
    k = np.where((fm[:-1] > 1) & (fm[1:] <= 1))[0][0]
    return f"{T[k] + (fm[k]-1)/(fm[k]-fm[k+1])*(T[k+1]-T[k]):.4f}"
lines = ["nvac,T_where_peak_of_avg_map_equals_1,T_where_mean_of_config_peaks_equals_1"]
for nv in doses:
    fa = np.array([r["f"] for r in runs[nv]])
    lines.append(f"{nv},{cross1(fa.mean(axis=0).max(axis=1))},{cross1(fa.max(axis=2).mean(axis=0))}")
write("witness_T.csv", lines)

# F: averaged map against per configuration statistics
def hnorm(f):
    p = np.clip(f, 0, None); p = p / p.sum(); p = p[p > 0]
    return -(p * np.log(p)).sum() / np.log(len(f))
lines = ["nvac,T,peak_of_avg_map,mean_of_config_peaks,Hnorm_of_avg_map,mean_of_config_Hnorm,mean_f_over_q"]
for nv in doses:
    fa = np.array([r["f"] for r in runs[nv]]); fbar = fa.mean(axis=0)
    for t in range(nT):
        lines.append(f"{nv},{T[t]:g},{fbar[t].max():.5f},{fa[:,t].max(axis=1).mean():.5f},{hnorm(fbar[t]):.5f},"
                     f"{np.mean([hnorm(x[t]) for x in fa]):.5f},{fbar[t].mean():.5f}")
write("averaged.csv", lines)

# E: energy placement of the QFI at q*
wc = 0.5 * (q_edges[1:] + q_edges[:-1])
lines = ["nvac,T,energy_centroid_J,share_below_0.5J"]
for nv in doses:
    Q = mean(nv, "Q")[:, iq]
    if not Q.any():
        continue
    for t in range(nT):
        lines.append(f"{nv},{T[t]:g},{(Q[t]*wc).sum()/Q[t].sum():.4f},{Q[t][wc<0.5].sum()/Q[t].sum():.4f}")
write("energy.csv", lines if len(lines) > 1 else ["no spectra stored in these files"])
