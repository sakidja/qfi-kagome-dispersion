#!/usr/bin/env python3
"""
Ground state bond map of the kagome Heisenberg antiferromagnet with vacancies,
for clusters too large for the full temperature run (27 sites and up).

Sparse Lanczos in the lowest Sz sector (Sz = 0 for an even spin count, 1/2 for
odd). Output: energy, gap inside the sector, G_ij = 4 <Sz_i Sz_j> for every
pair, <Sz_i>, and the bond table sorted by steps from the hole:
   (1,1) locked pair, (1,2) from a locked node into its other triangle,
   (2,2), (2,3), (3,3), (3,4), ...
At T = 0 the state is pure, so the QFI equals the full correlation: the bond
quantum share is 1 and G itself is the QFI bond map.
Open system reading: singlet fraction of a bond p_s = 1/4 - 3G/4 (exact when
the ground state has total spin 0 or 1/2); p_s > 1/2 means the two spins are
entangled with each other as a pair.

Usage:  python kagome_gs.py --L1 3 --L2 3 --vac 0 --out gs
        python kagome_gs.py --L1 3 --L2 3 --out gs            (clean reference)
AI (LLM) assistance (Claude, Anthropic) was used in preparing this code.
"""
import argparse, os, time
from collections import defaultdict, deque
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from qfi_kagome_ed import kagome_cluster


def sector(N, nup, chunk=1 << 24):
    """all N bit integers with nup bits set, ascending"""
    out = []
    m1, m2, m4 = 0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F
    for a in range(0, 1 << N, chunk):
        x = np.arange(a, min(a + chunk, 1 << N), dtype=np.int64)
        c = x - ((x >> 1) & m1); c = (c & m2) + ((c >> 2) & m2); c = (c + (c >> 4)) & m4
        c = (c * 0x0101010101010101) >> 56
        out.append(x[c == nup])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L1", type=int, required=True); ap.add_argument("--L2", type=int, required=True)
    ap.add_argument("--vac", type=int, nargs="*", default=[], help="site numbers to remove")
    ap.add_argument("--J", type=float, default=1.0); ap.add_argument("--nev", type=int, default=4)
    ap.add_argument("--out", default="gs")
    ap.add_argument("--no_check", action="store_true", help="skip the second solve that tests for degeneracy")
    a = ap.parse_args()
    t0 = time.time(); os.makedirs(a.out, exist_ok=True)
    pos0, bonds0, _ = kagome_cluster(a.L1, a.L2); N0 = len(pos0)
    vac = sorted(a.vac); keep = [s for s in range(N0) if s not in vac]; new = {s: k for k, s in enumerate(keep)}
    bonds = [(new[i], new[j]) for i, j in bonds0 if i in new and j in new]
    N = len(keep); nup = (N + 1) // 2
    tag = f"L{a.L1}x{a.L2}_gs_v" + ("_".join(map(str, vac)) if vac else "none")
    print(f"{tag}: {N0} sites, {N} spins, {len(bonds)} bonds, sector nup = {nup}", flush=True)
    st = sector(N, nup); D = len(st)
    print(f"dimension {D}  ({time.time()-t0:.0f}s)", flush=True)
    idx = np.int32 if D < 2**31 - 1 else np.int64
    diag = np.zeros(D); rows, cols = [], []
    for i, j in bonds:
        bi, bj = (st >> i) & 1, (st >> j) & 1
        anti = bi != bj
        diag += np.where(anti, -0.25, 0.25) * a.J
        src = np.nonzero(anti)[0]
        rows.append(np.searchsorted(st, st[src] ^ ((1 << i) | (1 << j))).astype(idx)); cols.append(src.astype(idx))
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    H = sp.coo_matrix((np.full(len(rows), 0.5 * a.J), (rows, cols)), shape=(D, D)).tocsr()
    del rows, cols
    H = H + sp.diags(diag)
    print(f"matrix built, {H.nnz} entries  ({time.time()-t0:.0f}s)", flush=True)
    if D <= 4000:
        E, V = np.linalg.eigh(H.toarray())
    else:
        E, V = spla.eigsh(H, k=min(a.nev, D - 2), which="SA", tol=1e-10)
    o = np.argsort(E); E, V = E[o], V[:, o]
    deg = int(np.sum(E - E[0] < 1e-7))
    gap = E[deg] - E[0] if deg < len(E) else float("nan")
    print(f"E0 = {E[0]:.10f}   E0 per spin = {E[0]/N:.6f}   degeneracy found = {deg}   gap in sector = {gap:.6f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)
    if deg == len(E):
        print("WARNING: all computed states are degenerate, raise --nev")
    # A Lanczos solver can return a single state of a degenerate level. Solve once more
    # from a different start vector; a different ground vector reveals the degeneracy.
    if D > 4000 and deg == 1 and not a.no_check:
        rng = np.random.default_rng(7)
        E2, V2 = spla.eigsh(H, k=1, which="SA", tol=1e-10, v0=rng.standard_normal(D))
        ov = abs(V2[:, 0] @ V[:, 0])
        print(f"degeneracy check: second solve E0 = {E2[0]:.10f}, overlap with first = {ov:.6f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
        if ov < 0.9999 and abs(E2[0] - E[0]) < 1e-7:
            w2 = V2[:, 0] - (V2[:, 0] @ V[:, 0]) * V[:, 0]; w2 /= np.linalg.norm(w2)
            V = np.column_stack([V[:, 0], w2, V[:, 1:]]); E = np.concatenate([[E[0], E[0]], E[1:]]); deg = 2
            print("ground level is degenerate (at least twofold): correlations are averaged over two states;"
                  " G_min and G_max are then meaningful only if the degeneracy is exactly two")
    # correlations, averaged over the degenerate ground states (the T -> 0 limit)
    w = (V[:, :deg] ** 2).sum(axis=1) / deg
    s = np.empty((N, D), dtype=np.float32)
    for k in range(N):
        s[k] = ((st >> k) & 1) - 0.5
    msz = s @ w
    G = 4.0 * (s * w.astype(np.float32)) @ s.T
    np.savez(os.path.join(a.out, tag + ".npz"), L=[a.L1, a.L2], vac=vac, keep=keep, pos=pos0[keep], bonds=bonds,
             E=E, deg=deg, G=G, mean_sz=msz)
    # bond table by steps from the hole
    lines = ["label,bonds,G_mean,G_min,G_max,singlet_fraction,pair_entangled"]
    if vac:
        nb = defaultdict(set)
        for i, j in bonds0:
            nb[i].add(j); nb[j].add(i)
        dist = {v: 0 for v in vac}; q = deque(vac)
        while q:
            u = q.popleft()
            for x in nb[u]:
                if x not in dist:
                    dist[x] = dist[u] + 1; q.append(x)
        per = defaultdict(list)
        for i, j in bonds0:
            if i in new and j in new:
                lab = tuple(sorted((dist[i], dist[j])))
                if lab == (1, 1) and not any(i in nb[v] and j in nb[v] for v in vac):
                    lab = (1, 1, "between holes")
                per[lab].append(G[new[i], new[j]])
        for lab in sorted(per, key=lambda l: (sum(l[:2]), len(l), l[:2])):
            g = np.array(per[lab]); extra = " between holes" if len(lab) > 2 else ""
            lines.append(f"\"({lab[0]},{lab[1]}){extra}\",{len(g)},{g.mean():.4f},{g.min():.4f},{g.max():.4f},{0.25-0.75*g.mean():.4f},{'yes' if 0.25-0.75*g.mean()>0.5 else 'no'}")
        sz_by = defaultdict(list)
        for k, site in enumerate(keep):
            sz_by[dist[site]].append(msz[k])
        lines.append("steps_from_hole,sites,mean_Sz")
        for dd in sorted(sz_by):
            lines.append(f"{dd},{len(sz_by[dd])},{np.mean(sz_by[dd]):.4f}")
    else:
        g = np.array([G[i, j] for i, j in bonds])
        lines.append(f"\"clean\",{len(g)},{g.mean():.4f},{g.min():.4f},{g.max():.4f},{0.25-0.75*g.mean():.4f},{'yes' if 0.25-0.75*g.mean()>0.5 else 'no'}")
    open(os.path.join(a.out, tag + "_shells.csv"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print(f"done ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
