#!/usr/bin/env python3
"""Tutorial step 1: two spins and three spins, by exact diagonalization.

Checks the hand results of the paper (Section 1.1, exercises 1 and 2):
  dimer    : QFI / (4 Var) of O = Sz_1 - Sz_2 equals tanh^2(J / 2T)
  triangle : G = -1/3 on every bond and singlet fraction 1/2 at low temperature
  triangle with one corner removed : G = -1, singlet fraction 1
Definitions as in the paper:
  G_ij = 4 (<Sz_i Sz_j> - <Sz_i><Sz_j>)
  F_ij = 2 sum_{m,n} (p_m - p_n)^2 / (p_m + p_n) <m|Sz_i|n><n|Sz_j|m>
  singlet fraction p_s = 1/4 - 3 G / 4
Writes step1_dimer.csv and step1_triangle.csv in the output folder.
"""
import argparse, os
import numpy as np

sx = np.array([[0, .5], [.5, 0]]); sy = np.array([[0, -.5j], [.5j, 0]]); sz = np.array([[.5, 0], [0, -.5]])

def site_op(op, i, n):
    out = np.array([[1.0]])
    for k in range(n):
        out = np.kron(out, op if k == i else np.eye(2))
    return out

def heisenberg(n, bonds, J=1.0):
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i, j in bonds:
        for s in (sx, sy, sz):
            H += J * site_op(s, i, n) @ site_op(s, j, n)
    return H

def F_and_G(H, n, T):
    """QFI matrix F and equal time matrix G of the thermal state at temperature T."""
    E, V = np.linalg.eigh(H)
    p = np.exp(-(E - E.min()) / T); p /= p.sum()
    Sz = [V.conj().T @ site_op(sz, i, n) @ V for i in range(n)]
    P, Q = np.meshgrid(p, p, indexing="ij")
    W = np.where(P + Q > 1e-300, 2 * (P - Q) ** 2 / np.maximum(P + Q, 1e-300), 0.0)
    F = np.zeros((n, n)); G = np.zeros((n, n))
    mean = [float(np.real(np.sum(p * np.diag(S)))) for S in Sz]
    for i in range(n):
        for j in range(n):
            F[i, j] = np.real(np.sum(W * Sz[i] * Sz[j].T))
            G[i, j] = 4 * (np.real(np.sum(p * np.diag(Sz[i] @ Sz[j]))) - mean[i] * mean[j])
    return F, G

ap = argparse.ArgumentParser()
ap.add_argument("--T", type=float, nargs="+", default=[0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
ap.add_argument("--J", type=float, default=1.0)
ap.add_argument("--out", default="tutorial_out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

print("== dimer: O = Sz_1 - Sz_2")
print("T,QFI,4Var,ratio,tanh2(J/2T),f_per_site,G_bond,singlet_fraction")
rows = []
c = np.array([1.0, -1.0])
for T in a.T:
    F, G = F_and_G(heisenberg(2, [(0, 1)], a.J), 2, T)
    qfi = c @ F @ c; var4 = c @ G @ c
    r = (T, qfi, var4, qfi / var4, np.tanh(a.J / (2 * T)) ** 2, qfi / 2, G[0, 1], 0.25 - 0.75 * G[0, 1])
    rows.append(r); print(",".join(f"{x:.6f}" for x in r))
np.savetxt(os.path.join(a.out, "step1_dimer.csv"), rows, delimiter=",", fmt="%.8f", comments="",
           header="T,QFI,4Var,ratio,tanh2,f_per_site,G_bond,singlet_fraction")

print("\n== triangle: three spins, three bonds")
print("T,G_bond,singlet_fraction,minus_J_over_4T")
rows = []
for T in a.T:
    F, G = F_and_G(heisenberg(3, [(0, 1), (1, 2), (0, 2)], a.J), 3, T)
    g = (G[0, 1] + G[1, 2] + G[0, 2]) / 3
    r = (T, g, 0.25 - 0.75 * g, -a.J / (4 * T))
    rows.append(r); print(",".join(f"{x:.6f}" for x in r))
np.savetxt(os.path.join(a.out, "step1_triangle.csv"), rows, delimiter=",", fmt="%.8f", comments="",
           header="T,G_bond,singlet_fraction,minus_J_over_4T")

print("\nhand results: dimer ratio = tanh^2(J/2T); triangle G = -1/3, singlet fraction 1/2 at low T;")
print("              triangle with one corner removed = the dimer: G = -1, singlet fraction 1 at low T;")
print("              triangle at high T: G approaches -J/(4T)")
