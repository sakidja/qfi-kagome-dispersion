#!/usr/bin/env python3
"""
q resolved quantum Fisher information (QFI) of the spin 1/2 kagome Heisenberg
antiferromagnet with vacancies, by exact diagonalization in total Sz sectors.

For every configuration (cluster + vacancy set) the script stores, at each
temperature T:

  F_ij  = 2 sum_{m,n} (p_m - p_n)^2 / (p_m + p_n) <m|S_i^z|n><n|S_j^z|m>
          the QFI matrix of the single site operators (real symmetric),
  G_ij  = 4 ( <S_i^z S_j^z> - <S_i^z><S_j^z> )
          four times the connected equal time correlation (what diffuse
          diffraction sees), which bounds F from above (F <= G as forms),
  S(q,w) and the energy resolved QFI density at every commensurate q.

From F the analysis script builds, for any q,
  f(q) = (1/N) sum_ij F_ij cos(q.(r_i - r_j)) = (F_Q[C_q] + F_Q[S_q]) / N,
  C_q = sum_i cos(q.r_i) S_i^z ,  S_q = sum_i sin(q.r_i) S_i^z .
Witness: f(q) > k  excludes k producible states, so at least (k+1) spins are
entangled together (Hauke, Heyl, Tagliacozzo, Zoller, Nat. Phys. 12, 778 (2016);
the weighted operator bound follows from Cauchy Schwarz on each block).

Method notes
  * S^z_i is diagonal in the computational basis and conserves total Sz, so
    every matrix element lives inside one Sz sector; sectors are diagonalized
    one at a time and never coupled.
  * Weights are carried unnormalized, b_m = exp(-(E_m - E0)/T); F, G and the
    spectra are homogeneous of degree one in p, so the partition function Z is
    divided out at the end.
  * Only rows m with b_m > eps at the highest T are kept ("populated" set K).
    Pairs with both states outside K carry weight below eps and are dropped.
    Pairs (m in K, n outside K) are counted twice to include (n, m).

AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse
import itertools
import json
import os
import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ----------------------------------------------------------------------------
# lattice
# ----------------------------------------------------------------------------
def kagome_cluster(L1, L2):
    """Periodic L1 x L2 kagome cluster, nearest neighbour distance 1."""
    a1 = np.array([2.0, 0.0])
    a2 = np.array([1.0, np.sqrt(3.0)])
    sub = [np.zeros(2), a1 / 2, a2 / 2]
    idx, pos = {}, []
    for x in range(L1):
        for y in range(L2):
            for s in range(3):
                idx[(x, y, s)] = len(pos)
                pos.append(x * a1 + y * a2 + sub[s])

    def site(x, y, s):
        return idx[(x % L1, y % L2, s)]

    bonds = set()
    for x in range(L1):
        for y in range(L2):
            A, B, C = site(x, y, 0), site(x, y, 1), site(x, y, 2)
            # down triangle: B(R), C(R + a1 + (-a2)), A(R + a1)
            Bd, Cd, Ad = site(x, y, 1), site(x + 1, y - 1, 2), site(x + 1, y, 0)
            for i, j in [(A, B), (A, C), (B, C), (Bd, Cd), (Bd, Ad), (Cd, Ad)]:
                if i == j:
                    raise ValueError("cluster too small: self bond")
                bonds.add((min(i, j), max(i, j)))
    N = len(pos)
    if len(bonds) != 2 * N:
        raise ValueError(f"cluster {L1}x{L2}: {len(bonds)} bonds, expected {2*N}")
    A = np.array([a1, a2])
    return np.array(pos), sorted(bonds), A


def commensurate_q(L1, L2, A):
    """All q = (n1/L1) b1 + (n2/L2) b2 with n1 in [-L1, L1), n2 in [-L2, L2).
    This covers one full period (2 b1, 2 b2) of the kagome structure factor."""
    B = 2 * np.pi * np.linalg.inv(A).T          # rows b1, b2
    qs, labels = [], []
    for n1 in range(-L1, L1):
        for n2 in range(-L2, L2):
            qs.append((n1 / L1) * B[0] + (n2 / L2) * B[1])
            labels.append((n1, n2))
    return np.array(qs), np.array(labels), B


# ----------------------------------------------------------------------------
# Hilbert space and Hamiltonian
# ----------------------------------------------------------------------------
def sector_states(N, nup):
    st = np.array([sum(1 << k for k in c) for c in itertools.combinations(range(N), nup)],
                  dtype=np.int64)
    st.sort()
    return st


def sz_table(states, N):
    return (((states[:, None] >> np.arange(N)) & 1) - 0.5).astype(np.float64)


def hamiltonian_sparse(states, sz, bonds, J=1.0):
    D = len(states)
    diag = np.zeros(D)
    rows, cols = [], []
    for (i, j) in bonds:
        diag += J * sz[:, i] * sz[:, j]
        src = np.nonzero(sz[:, i] != sz[:, j])[0]
        dst = np.searchsorted(states, states[src] ^ ((1 << i) | (1 << j)))
        rows.append(dst)
        cols.append(src)
    if rows:
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.full(len(rows), 0.5 * J)
    else:
        rows = cols = np.zeros(0, dtype=np.int64)
        vals = np.zeros(0)
    H = sp.coo_matrix((vals, (rows, cols)), shape=(D, D)).tocsr()
    H = H + sp.diags(diag)
    return H


def lowest_energy(H):
    D = H.shape[0]
    if D <= 3000:
        return np.linalg.eigvalsh(H.toarray())[0]
    return spla.eigsh(H, k=1, which="SA", tol=1e-12)[0][0]


# ----------------------------------------------------------------------------
# one configuration
# ----------------------------------------------------------------------------
def run_config(pos_all, bonds_all, vac, Ts, qs, eps, J, omega_max, nbins,
               mem_gb, spectra, log):
    keep = np.array([i for i in range(len(pos_all)) if i not in set(vac)])
    relabel = {int(s): k for k, s in enumerate(keep)}
    pos = pos_all[keep]
    bonds = [(relabel[i], relabel[j]) for (i, j) in bonds_all
             if i in relabel and j in relabel]
    N = len(keep)
    Ts = np.asarray(Ts, float)
    nT, nq = len(Ts), len(qs)

    # phase factors at commensurate q for the present sites
    phase = qs @ pos.T                                   # (nq, N)
    cq, sq = np.cos(phase), np.sin(phase)

    # pass 1: global ground energy
    t0 = time.time()
    e0s = []
    for nup in range(N + 1):
        st = sector_states(N, nup)
        H = hamiltonian_sparse(st, sz_table(st, N), bonds, J)
        e0s.append(lowest_energy(H))
    E0 = min(e0s)
    cut = -Ts.max() * np.log(eps)                        # keep E - E0 < cut
    log(f"  N={N} bonds={len(bonds)} E0={E0:.10f} cutoff dE<{cut:.3f}  ({time.time()-t0:.1f}s)")

    F = np.zeros((nT, N, N))
    G2 = np.zeros((nT, N, N))     # unnormalized <SiSj>
    M1 = np.zeros((nT, N))        # unnormalized <Si>
    Z = np.zeros(nT)
    Kmax_info = []
    s_edges = np.linspace(-omega_max, omega_max, 2 * nbins + 1)
    q_edges = np.linspace(0.0, omega_max, nbins + 1)
    dws = s_edges[1] - s_edges[0]
    Sqw = np.zeros((nT, nq, 2 * nbins))
    Qqw = np.zeros((nT, nq, nbins))

    # pass 2: sector by sector
    for nup in range(N + 1):
        if e0s[nup] - E0 >= cut:
            # whole sector unpopulated at every T: contributes only to Z
            st = sector_states(N, nup)
            H = hamiltonian_sparse(st, sz_table(st, N), bonds, J).toarray()
            E = np.linalg.eigvalsh(H)
            for t, T in enumerate(Ts):
                Z[t] += np.exp(-(E - E0) / T).sum()
            continue
        t1 = time.time()
        st = sector_states(N, nup)
        sz = sz_table(st, N)
        D = len(st)
        H = hamiltonian_sparse(st, sz, bonds, J).toarray()
        if D > 1:
            E, V = sla.eigh(H, overwrite_a=True, check_finite=False, driver="evr")
        else:
            E, V = np.diag(H).copy(), np.ones((1, 1))
        del H
        dE = E - E0
        b = np.exp(-dE[None, :] / Ts[:, None])          # (nT, D)
        Z += b.sum(axis=1)
        K = int(np.searchsorted(dE, cut))               # E sorted ascending
        Kmax_info.append((nup, D, K))
        if K == 0:
            continue

        # diagonal density for <SiSj>, <Si>
        Vk2 = V[:, :K] ** 2                              # (D, K)
        rho = Vk2 @ b[:, :K].T                           # (D, nT)
        for t in range(nT):
            G2[t] += sz.T @ (sz * rho[:, t:t + 1])
            M1[t] += sz.T @ rho[:, t]
        del Vk2, rho

        # row blocks of the populated set
        Bsz = max(1, min(K, int(mem_gb * 1e9 / (8.0 * D * (N + 3)))))
        far = np.ones(D)
        far[K:] = 2.0                                    # (m in K, n outside K) counted twice
        for r0 in range(0, K, Bsz):
            r1 = min(K, r0 + Bsz)
            Bn = r1 - r0
            VB = V[:, r0:r1]
            Mst = np.empty((N, Bn, D))
            for i in range(N):
                Mst[i] = (VB * sz[:, i:i + 1]).T @ V
            Mf = Mst.reshape(N, Bn * D)
            om = E[None, :] - E[r0:r1, None]             # (Bn, D) = E_n - E_m
            if spectra:
                sbin = np.floor((om + omega_max) / dws).astype(np.int64)
                sok = (sbin >= 0) & (sbin < 2 * nbins)
                qbin = np.floor(np.abs(om) / (q_edges[1])).astype(np.int64)
                qok = qbin < nbins
            for t, T in enumerate(Ts):
                bm = b[t, r0:r1][:, None]
                bn = b[t][None, :]
                den = bm + bn
                W = np.zeros_like(om)
                np.divide(2.0 * (bm - bn) ** 2, den, out=W, where=den > 0)
                W *= far[None, :]
                Wf = W.reshape(-1)
                F[t] += (Mf * Wf) @ Mf.T
                if spectra:
                    BM = np.broadcast_to(bm, om.shape)
                    for iq in range(nq):
                        Mc = np.tensordot(cq[iq], Mst, axes=(0, 0))
                        Ms = np.tensordot(sq[iq], Mst, axes=(0, 0))
                        A2 = Mc * Mc + Ms * Ms               # |<n|O_q|m>|^2, O_q = C + iS
                        Sqw[t, iq] += np.bincount(sbin[sok], weights=(BM * A2)[sok],
                                                  minlength=2 * nbins)
                        Qqw[t, iq] += np.bincount(qbin[qok], weights=(W * A2)[qok],
                                                  minlength=nbins)
            del Mst, Mf
        log(f"  sector nup={nup:2d} D={D:6d} K={K:6d}  {time.time()-t1:.1f}s")

    F /= Z[:, None, None]
    mean = M1 / Z[:, None]
    G = 4.0 * (G2 / Z[:, None, None] - mean[:, :, None] * mean[:, None, :])
    Sqw /= (Z[:, None, None] * dws)                      # density per unit energy
    Qqw /= Z[:, None, None]                              # QFI per bin (sum over bins = F_Q)
    return dict(N=N, keep=keep, pos=pos, bonds=np.array(bonds), E0=E0, Z=Z, F=F, G=G,
                mean_sz=mean, Sqw=Sqw, Qqw=Qqw, s_edges=s_edges, q_edges=q_edges,
                sector_info=np.array(Kmax_info))


# ----------------------------------------------------------------------------
# fast path (no spectra): only the populated eigenvectors are computed
# ----------------------------------------------------------------------------
def run_config_fast(pos_all, bonds_all, vac, Ts, qs, eps, J, omega_max, nbins, mem_gb, log, ckpt):
    """
    Same F, G, <Sz> and Z as run_config(spectra=False), obtained with three exact shortcuts.

    1. Completeness. For a populated state m and an unpopulated state n the QFI weight is 4 b_m, so
         sum over unpopulated n of <m|Si|n><n|Sj|m>  =  <m|Si Sj|m>  -  sum over populated n.
       Only the K populated eigenvectors of each sector are therefore needed (partial
       diagonalization), and the matrix elements are K by K instead of K by D.
    2. Spin flip symmetry. Without a field the sectors nup and N - nup give identical F and
       <Si Sj> and opposite <Si>, so only nup <= N/2 is computed and counted twice.
    3. Checkpoint. The sums are saved after every sector; a restarted job skips finished sectors.

    States with Boltzmann weight below eps at the highest T are left out of Z as well (they were
    already left out of the populated set); the change in Z is below D * eps.
    """
    keep = np.array([i for i in range(len(pos_all)) if i not in set(vac)])
    relabel = {int(s_): k for k, s_ in enumerate(keep)}
    pos = pos_all[keep]
    bonds = [(relabel[i], relabel[j]) for (i, j) in bonds_all if i in relabel and j in relabel]
    N = len(keep)
    Ts = np.asarray(Ts, float)
    nT, nq = len(Ts), len(qs)
    t0 = time.time()
    done, st_ = [], None
    if ckpt and os.path.exists(ckpt):
        st_ = np.load(ckpt)
        if int(st_["N"]) == N and np.allclose(st_["Ts"], Ts) and list(st_["vac"]) == list(vac):
            done = [int(x) for x in st_["done"]]
        else:
            st_ = None
    if st_ is not None:
        E0, e0s = float(st_["E0"]), list(st_["e0s"])
        F, G2, M1, Z = st_["F"].copy(), st_["G2"].copy(), st_["M1"].copy(), st_["Z"].copy()
        info = [tuple(r) for r in st_["info"]]
        log(f"  restart from checkpoint, sectors done: {done}")
    else:
        e0s = []
        for nup in range(N // 2 + 1):
            st = sector_states(N, nup)
            e0s.append(lowest_energy(hamiltonian_sparse(st, sz_table(st, N), bonds, J)))
        E0 = min(e0s)
        F = np.zeros((nT, N, N)); G2 = np.zeros((nT, N, N)); M1 = np.zeros((nT, N)); Z = np.zeros(nT)
        info = []
    cut = -Ts.max() * np.log(eps)
    log(f"  N={N} bonds={len(bonds)} E0={E0:.10f} cutoff dE<{cut:.3f}  fast path  ({time.time()-t0:.1f}s)")

    for nup in range(N // 2 + 1):
        if nup in done:
            continue
        t1 = time.time()
        wsec = 1.0 if 2 * nup == N else 2.0               # nup and N - nup together
        st = sector_states(N, nup)
        sz = sz_table(st, N)
        D = len(st)
        H = hamiltonian_sparse(st, sz, bonds, J).toarray()
        if e0s[nup] - E0 >= cut:
            K = 0
        elif D <= 2:
            E, V = np.linalg.eigh(H); K = int(np.searchsorted(E - E0, cut)); E, V = E[:K], V[:, :K]
        else:
            E, V = sla.eigh(H, overwrite_a=True, check_finite=False, driver="evr",
                            subset_by_value=(-np.inf, E0 + cut))
            K = len(E)
        del H
        info.append((nup, D, K))
        if K > 0:
            b = np.exp(-(E[None, :] - E0) / Ts[:, None])     # (nT, K)
            Z += wsec * b.sum(axis=1)
            rho = (V ** 2) @ b.T                             # (D, nT)
            G2s = np.empty((nT, N, N))
            for t in range(nT):
                G2s[t] = sz.T @ (sz * rho[:, t:t + 1])
                if wsec == 1.0:
                    M1[t] += sz.T @ rho[:, t]                # paired sectors cancel in <Si>
            G2 += wsec * G2s
            del rho
            Bsz = max(1, min(K, int(mem_gb * 1e9 / (8.0 * K * (N + 3)))))
            Fs = 4.0 * G2s
            for r0 in range(0, K, Bsz):
                r1 = min(K, r0 + Bsz); Bn = r1 - r0
                VB = V[:, r0:r1]
                Mst = np.empty((N, Bn, K))
                for i in range(N):
                    Mst[i] = (VB * sz[:, i:i + 1]).T @ V
                Mf = Mst.reshape(N, Bn * K)
                for t in range(nT):
                    bm = b[t, r0:r1][:, None]; bn = b[t][None, :]
                    W = 2.0 * (bm - bn) ** 2 / (bm + bn) - 4.0 * bm
                    Fs[t] += (Mf * W.reshape(-1)) @ Mf.T
                del Mst, Mf
            F += wsec * Fs
            del V
        done.append(nup)
        if ckpt:
            np.savez(ckpt, N=N, Ts=Ts, vac=np.array(vac, dtype=int), E0=E0, e0s=np.array(e0s), done=np.array(done),
                     F=F, G2=G2, M1=M1, Z=Z, info=np.array(info))
        log(f"  sector nup={nup:2d}" + (f" (and {N-nup:2d})" if 2 * nup != N else "         ") + f" D={D:6d} K={K:6d}  {time.time()-t1:.1f}s")

    Fo = F / Z[:, None, None]
    mean = M1 / Z[:, None]
    G = 4.0 * (G2 / Z[:, None, None] - mean[:, :, None] * mean[:, None, :])
    s_edges = np.linspace(-omega_max, omega_max, 2 * nbins + 1)
    q_edges = np.linspace(0.0, omega_max, nbins + 1)
    return dict(N=N, keep=keep, pos=pos, bonds=np.array(bonds), E0=E0, Z=Z, F=Fo, G=G, mean_sz=mean,
                Sqw=np.zeros((nT, nq, 2 * nbins)), Qqw=np.zeros((nT, nq, nbins)), s_edges=s_edges,
                q_edges=q_edges, sector_info=np.array(info))



# ----------------------------------------------------------------------------
# checks on the stored result
# ----------------------------------------------------------------------------
def self_check(res, qs, log):
    pos = res["pos"]
    N = res["N"]
    d = pos[:, None, :] - pos[None, :, :]
    cosm = np.cos(np.einsum("qk,ijk->qij", qs, d))
    f = np.einsum("tij,qij->tq", res["F"], cosm) / N
    g = np.einsum("tij,qij->tq", res["G"], cosm) / N
    worst = (f - g).max()
    log(f"  check F <= G: max_q (f - g) = {worst:.3e}  (should be <= ~1e-10)")
    if res["Qqw"].any():
        fq_from_w = res["Qqw"].sum(axis=2) / N
        log(f"  check energy sum of QFI vs f(q): max |diff| = {np.abs(fq_from_w - f).max():.3e}"
            "  (small unless omega_max is below the bandwidth)")
    ok = g[0] > 1e-6 * g[0].max()        # q = 0 is total Sz: conserved, f = 0 exactly
    lo = f[0][ok] / g[0][ok]
    log(f"  check lowest T: min_q f/g = {lo.min():.4f} over q with g > 0 "
        "(approaches 1 as T -> 0; a degenerate ground multiplet keeps it below 1)")
    return f, g


def vacancy_sets(N, nvac, nconf, seed):
    if nvac == 0:
        return [()]
    rng = np.random.default_rng(seed)
    total = 1
    for k in range(nvac):
        total = total * (N - k) // (k + 1)
    if nconf >= total:
        return [tuple(c) for c in itertools.combinations(range(N), nvac)]
    seen, out = set(), []
    while len(out) < nconf:
        c = tuple(sorted(rng.choice(N, nvac, replace=False).tolist()))
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--L1", type=int, default=2)
    ap.add_argument("--L2", type=int, default=3)
    ap.add_argument("--nvac", type=int, default=0, help="vacancies per configuration")
    ap.add_argument("--nconf", type=int, default=1, help="configurations for this dose")
    ap.add_argument("--conf", type=int, default=None,
                    help="run only this configuration index (for job arrays)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--T", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3],
                    help="temperatures in units of J")
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--eps", type=float, default=1e-8,
                    help="Boltzmann weight cutoff at the highest T")
    ap.add_argument("--omega_max", type=float, default=6.0, help="energy window (J)")
    ap.add_argument("--nbins", type=int, default=120)
    ap.add_argument("--no_spectra", action="store_true")
    ap.add_argument("--mem_gb", type=float, default=8.0, help="memory budget per row block")
    ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true",
                    help="with --no_spectra: partial diagonalization, spin flip symmetry and a checkpoint "
                         "after every sector (a restarted job continues where it stopped)")
    a = ap.parse_args()
    if a.fast and not a.no_spectra:
        ap.error("--fast needs --no_spectra (the spectra need all eigenstates)")

    os.makedirs(a.out, exist_ok=True)
    pos, bonds, A = kagome_cluster(a.L1, a.L2)
    qs, qlab, B = commensurate_q(a.L1, a.L2, A)
    N0 = len(pos)
    confs = vacancy_sets(N0, a.nvac, a.nconf, a.seed)
    which = range(len(confs)) if a.conf is None else [a.conf]
    tag = f"L{a.L1}x{a.L2}_nv{a.nvac}"
    logf = open(os.path.join(a.out, f"{tag}.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    log(f"=== kagome {a.L1}x{a.L2}, N0={N0}, nvac={a.nvac}, configs={len(confs)}, T={a.T}")
    for c in which:
        vac = confs[c]
        log(f"config {c}: vacancies {list(vac)}")
        t0 = time.time()
        if a.fast:
            res = run_config_fast(pos, bonds, vac, a.T, qs, a.eps, a.J, a.omega_max, a.nbins, a.mem_gb, log,
                                  os.path.join(a.out, f"{tag}_c{c}_ckpt.npz"))
        else:
            res = run_config(pos, bonds, vac, a.T, qs, a.eps, a.J, a.omega_max, a.nbins,
                             a.mem_gb, not a.no_spectra, log)
        self_check(res, qs, log)
        fn = os.path.join(a.out, f"{tag}_c{c}.npz")
        np.savez_compressed(fn, T=np.array(a.T), qs=qs, qlab=qlab, B=B, N0=N0,
                            vac=np.array(vac, dtype=int), L=np.array([a.L1, a.L2]),
                            eps=a.eps, J=a.J, **res)
        log(f"  saved {fn}  total {time.time()-t0:.1f}s")
    with open(os.path.join(a.out, f"{tag}_args.json"), "w") as fh:
        json.dump(vars(a), fh, indent=1)


if __name__ == "__main__":
    main()
