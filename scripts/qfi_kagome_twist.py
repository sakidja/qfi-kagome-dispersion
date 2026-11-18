#!/usr/bin/env python3
"""
Twisted boundary version of qfi_kagome_ed.py: finer wave vector resolution
from the same clusters.

A twist (theta1, theta2) around the two periodic directions is put on the
lattice in the uniform gauge: every spin flip term carries a phase,

  H = J sum_<ij> [ Sz_i Sz_j + (1/2)( e^{i kappa.d_ij} S+_i S-_j + h.c. ) ],

with d_ij = r_j - r_i the bond vector and kappa.(L1 a1) = theta1,
kappa.(L2 a2) = theta2. In this gauge the transverse spin operator
sum_j e^{i q.r_j} S+_j at a commensurate q carries the physical wave vector

  Q = q + kappa.

So a grid of nt1 x nt2 twists gives nt1 x nt2 times as many Q points as the
cluster alone. The transverse channel is what is computed here; at zero twist
it equals the Sz channel of qfi_kagome_ed.py exactly (spin rotation symmetry),
which is one of the checks in check_twist.py.

Stored per (configuration, twist), for each temperature:
  F   (N x N, complex Hermitian): f(Q) = (1/N) c^dag F c, c_j = e^{i q.r_j},
      the QFI of O = (1/2) sum_j (c_j S+_j + h.c.), per spin.
  G   same form for 4 Var(O): the equal time correlation, g(Q) >= f(Q).
  Sqw (1/2) S^{-+}(Q, w), equal to S^{zz}(Q, w) for a spin rotation
      invariant system; Qqw the energy resolved QFI (sum over bins = f N).

S+ changes total Sz by one, so matrix elements run between adjacent Sz
sectors; both sectors are diagonalized (complex Hermitian) and kept in memory.

AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse
import os
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from qfi_kagome_ed import kagome_cluster, commensurate_q, sector_states, sz_table, vacancy_sets


def bond_vectors(pos, bonds, A, L1, L2):
    """d_ij = r_j - r_i for each bond, as the unit length periodic image."""
    out = []
    for i, j in bonds:
        found = []
        for n1 in (-1, 0, 1):
            for n2 in (-1, 0, 1):
                d = pos[j] - pos[i] + n1 * L1 * A[0] + n2 * L2 * A[1]
                if abs(np.linalg.norm(d) - 1.0) < 1e-9:
                    found.append(d)
        if len(found) != 1:
            raise ValueError("bond image not unique; cluster too small for a twist")
        out.append(found[0])
    return np.array(out)


def hamiltonian(states, sz, bonds, dvec, kappa, J):
    D = len(states)
    diag = np.zeros(D)
    rows, cols, vals = [], [], []
    for (i, j), d in zip(bonds, dvec):
        diag += J * sz[:, i] * sz[:, j]
        src = np.nonzero(sz[:, i] != sz[:, j])[0]
        dst = np.searchsorted(states, states[src] ^ ((1 << i) | (1 << j)))
        sgn = np.where(sz[src, i] < 0, 1.0, -1.0)      # i down, j up: S+_i S-_j, phase +kappa.d
        rows.append(dst); cols.append(src)
        vals.append(0.5 * J * np.exp(1j * sgn * float(kappa @ d)))
    if rows:
        H = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(D, D)).tocsr()
    else:
        H = sp.csr_matrix((D, D), dtype=complex)
    return H + sp.diags(diag.astype(complex))


def lowest_energy(H):
    if H.shape[0] <= 3000:
        return np.linalg.eigvalsh(H.toarray())[0]
    return spla.eigsh(H, k=1, which="SA", tol=1e-12)[0][0].real


def gather(P, Er, Ec, edges):
    """Bin P[q, r, c] over rows by the energy Er[r] - Ec[c] into `edges`.
    Rows are in ascending energy, so each bin is a contiguous row range.
    Returns (nq, nbins, C)."""
    nq, R, C = P.shape
    cs = np.zeros((nq, R + 1, C))
    np.cumsum(P, axis=1, out=cs[:, 1:, :])
    idx = np.searchsorted(Er, Ec[None, :] + edges[:, None])          # (nb+1, C)
    g = cs[:, idx, np.arange(C)[None, :]]                            # (nq, nb+1, C)
    return np.diff(g, axis=1)


def run_twist(pos_all, bonds_all, dvec_all, vac, kappa, Ts, qs, eps, J, omega_max, nbins,
              mem_gb, spectra, log):
    keep = np.array([i for i in range(len(pos_all)) if i not in set(vac)])
    relabel = {int(s): k for k, s in enumerate(keep)}
    pos = pos_all[keep]
    sel = [k for k, (i, j) in enumerate(bonds_all) if i in relabel and j in relabel]
    bonds = [(relabel[bonds_all[k][0]], relabel[bonds_all[k][1]]) for k in sel]
    dvec = dvec_all[sel]
    N = len(keep)
    Ts = np.asarray(Ts, float)
    nT, nq = len(Ts), len(qs)
    cph = np.exp(1j * qs @ pos.T)                         # (nq, N), c_j = e^{i q.r_j}

    t0 = time.time()
    e0s = []
    for nup in range(N + 1):
        st = sector_states(N, nup)
        e0s.append(lowest_energy(hamiltonian(st, sz_table(st, N), bonds, dvec, kappa, J)))
    E0 = min(e0s)
    cut = -Ts.max() * np.log(eps)
    live = [e - E0 < cut for e in e0s]
    log(f"  N={N} bonds={len(bonds)} E0={E0:.10f} cutoff dE<{cut:.3f}  ({time.time()-t0:.1f}s)")

    Fm = np.zeros((nT, N, N), complex)
    Gm = np.zeros((nT, N, N), complex)
    Z = np.zeros(nT)
    s_edges = np.linspace(-omega_max, omega_max, 2 * nbins + 1)
    dws = s_edges[1] - s_edges[0]
    Sqw = np.zeros((nT, nq, 2 * nbins))
    Qsg = np.zeros((nT, nq, 2 * nbins))

    def weights(br, bc):                                  # br (nT,R), bc (nT,C) -> (nT,R,C)
        den = br[:, :, None] + bc[:, None, :]
        W = np.zeros_like(den)
        np.divide(2.0 * (br[:, :, None] - bc[:, None, :]) ** 2, den, out=W, where=den > 0)
        return W

    def pair(lo, hi):
        """lo: sector nup, hi: sector nup+1. A_i[n, m] = <n|S+_i|m>."""
        src, dst = [], []
        for i in range(N):
            s = np.nonzero(((lo["st"] >> i) & 1) == 0)[0]
            src.append(s)
            dst.append(np.searchsorted(hi["st"], lo["st"][s] | (1 << i)))
        for forward in (True, False):
            if forward:      # columns: populated states m of lo; rows: all states n of hi
                col, row, r0 = lo, hi, 0
            else:            # columns: populated states n of hi; rows: unpopulated m of lo
                col, row, r0 = hi, lo, lo["K"]
            R, Kc = len(row["E"]) - r0, col["K"]
            if R <= 0 or Kc == 0:
                continue
            Er = row["E"][r0:]
            Cb = max(1, min(Kc, int(mem_gb * 1e9 / (16.0 * R * (N + 2 * nq + 4)))))
            for c0 in range(0, Kc, Cb):
                c1 = min(Kc, c0 + Cb)
                A = np.empty((N, R, c1 - c0), complex)
                for i in range(N):
                    if forward:
                        A[i] = hi["V"][dst[i], :].conj().T @ lo["V"][src[i], c0:c1]
                    else:
                        A[i] = lo["V"][src[i], r0:].conj().T @ hi["V"][dst[i], c0:c1]
                Af = A.reshape(N, -1)
                Ec = col["E"][c0:c1]
                br, bc = row["b"][:, r0:], col["b"][:, c0:c1]
                W = weights(br, bc)
                for t in range(nT):
                    w = W[t].reshape(-1)
                    if forward:
                        Fm[t] += 0.5 * (Af.conj() * w) @ Af.T
                        wb = np.broadcast_to(bc[t][None, :], (R, c1 - c0)).reshape(-1)
                        Gm[t] += 2.0 * (Af.conj() * wb) @ Af.T
                    else:
                        Fm[t] += 0.5 * (Af * w) @ Af.conj().T
                if spectra:
                    Aq = np.tensordot(cph if forward else cph.conj(), A, axes=(1, 0))   # (nq,R,C)
                    P = Aq.real ** 2 + Aq.imag ** 2
                    del Aq
                    if forward:
                        gs = gather(P, Er, Ec, s_edges)                                 # (nq,2nb,C)
                        Sqw[:] += 0.5 * np.einsum("qbc,tc->tqb", gs, bc)
                        del gs
                    for t in range(nT):
                        Qsg[t] += 0.5 * gather(P * W[t][None], Er, Ec, s_edges).sum(axis=2)
                    del P
                del A, Af, W

    prev = None
    for nup in range(N + 1):
        need = live[nup] or (nup > 0 and live[nup - 1]) or (nup < N and live[nup + 1])
        st = sector_states(N, nup)
        H = hamiltonian(st, sz_table(st, N), bonds, dvec, kappa, J).toarray()
        t1 = time.time()
        if not need:
            E = np.linalg.eigvalsh(H)
            Z += np.exp(-(E[None, :] - E0) / Ts[:, None]).sum(axis=1)
            prev = None
            continue
        if len(st) > 1:
            E, V = sla.eigh(H, overwrite_a=True, check_finite=False, driver="evr")
        else:
            E, V = np.array([H[0, 0].real]), np.ones((1, 1), complex)
        del H
        b = np.exp(-(E[None, :] - E0) / Ts[:, None])
        Z += b.sum(axis=1)
        cur = dict(st=st, E=E, V=V, b=b, K=int(np.searchsorted(E - E0, cut)))
        if prev is not None and (prev["K"] > 0 or cur["K"] > 0):
            pair(prev, cur)
        log(f"  sector nup={nup:2d} D={len(st):6d} K={cur['K']:6d}  {time.time()-t1:.1f}s")
        prev = cur

    Fm /= Z[:, None, None]
    Gm /= Z[:, None, None]
    Sqw /= (Z[:, None, None] * dws)
    Qsg /= Z[:, None, None]
    Qqw = Qsg[:, :, nbins:] + Qsg[:, :, :nbins][:, :, ::-1]          # fold onto |w|
    return dict(N=N, keep=keep, pos=pos, bonds=np.array(bonds), E0=E0, Z=Z, F=Fm, G=Gm,
                Sqw=Sqw, Qqw=Qqw, s_edges=s_edges, q_edges=s_edges[nbins:])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L1", type=int, default=2)
    ap.add_argument("--L2", type=int, default=3)
    ap.add_argument("--nvac", type=int, default=0)
    ap.add_argument("--nconf", type=int, default=1)
    ap.add_argument("--conf", type=int, default=None, help="run only this configuration")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nt", type=int, nargs=2, default=[4, 4], help="twist grid nt1 nt2")
    ap.add_argument("--twist", type=int, nargs=2, default=None,
                    help="run only twist index t1 t2 (for job arrays)")
    ap.add_argument("--theta", type=float, nargs=2, default=None,
                    help="explicit twist angles in units of 2 pi (testing)")
    ap.add_argument("--T", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3])
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8)
    ap.add_argument("--omega_max", type=float, default=6.0)
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--no_spectra", action="store_true")
    ap.add_argument("--mem_gb", type=float, default=8.0)
    ap.add_argument("--out", default="results_twist")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    pos, bonds, A = kagome_cluster(a.L1, a.L2)
    dvec = bond_vectors(pos, bonds, A, a.L1, a.L2)
    qs, qlab, B = commensurate_q(a.L1, a.L2, A)
    confs = vacancy_sets(len(pos), a.nvac, a.nconf, a.seed)
    which = range(len(confs)) if a.conf is None else [a.conf]
    if a.theta is not None:
        twists = [("x", "x", a.theta[0], a.theta[1])]
    elif a.twist is not None:
        twists = [(a.twist[0], a.twist[1], a.twist[0] / a.nt[0], a.twist[1] / a.nt[1])]
    else:
        twists = [(t1, t2, t1 / a.nt[0], t2 / a.nt[1]) for t1 in range(a.nt[0]) for t2 in range(a.nt[1])]
    tag = f"L{a.L1}x{a.L2}_nv{a.nvac}"
    logf = open(os.path.join(a.out, f"{tag}.log"), "a")

    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()

    for c in which:
        for (t1, t2, th1, th2) in twists:
            kappa = (th1 / a.L1) * B[0] + (th2 / a.L2) * B[1]
            log(f"config {c}: vacancies {list(confs[c])}  twist ({th1:g}, {th2:g}) x 2pi")
            t0 = time.time()
            res = run_twist(pos, bonds, dvec, confs[c], kappa, a.T, qs, a.eps, a.J,
                            a.omega_max, a.nbins, a.mem_gb, not a.no_spectra, log)
            N = res["N"]
            cph = np.exp(1j * qs @ res["pos"].T)
            f = np.einsum("qi,tij,qj->tq", cph.conj(), res["F"], cph).real / N
            g = np.einsum("qi,tij,qj->tq", cph.conj(), res["G"], cph).real / N
            log(f"  check F <= G: max_Q (f - g) = {(f - g).max():.3e}   "
                f"sum rule mean_Q g = {g.mean(axis=1).round(6).tolist()}")
            fn = os.path.join(a.out, f"{tag}_c{c}_t{t1}_{t2}.npz")
            np.savez_compressed(fn, T=np.array(a.T), qs=qs, qlab=qlab, B=B, N0=len(pos),
                                vac=np.array(confs[c], dtype=int), L=np.array([a.L1, a.L2]),
                                kappa=kappa, theta=np.array([th1, th2]), nt=np.array(a.nt),
                                Q=qs + kappa, f=f, g=g, eps=a.eps, J=a.J, **res)
            log(f"  saved {fn}  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
