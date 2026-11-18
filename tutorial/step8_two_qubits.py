#!/usr/bin/env python3
"""Tutorial step 8: the bond correlation G on two qubits (exercise 8 of the paper).

Prepares the singlet (X on both qubits, H on the first, CNOT), applies a depolarizing error of
strength p, samples both qubits in the Z basis and estimates
  G = <Z1 Z2> - <Z1><Z2>          (equal to 4 times the connected <Sz Sz>)
  singlet fraction p_s = 1/4 - 3 G / 4
Exact values: G = -(1 - p), p_s = 1 - 3p/4, pair entangled for p < 2/3.
Pure numpy; the same circuit runs unchanged on any simulator or quantum processor.
Writes step8_two_qubits.csv (one row per p) and step8_counts.csv (raw counts) in the output folder.
"""
import argparse, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4, 0.6, 2 / 3, 0.8, 1.0])
ap.add_argument("--shots", type=int, default=20000)
ap.add_argument("--seed", type=int, default=1234)
ap.add_argument("--out", default="tutorial_out")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

X = np.array([[0, 1], [1, 0]]); Hd = np.array([[1, 1], [1, -1]]) / np.sqrt(2); I2 = np.eye(2)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
psi = np.zeros(4); psi[0] = 1.0                       # |00>, first qubit is the left bit
psi = CNOT @ np.kron(Hd, I2) @ np.kron(X, X) @ psi    # (|01> - |10>)/sqrt(2)
rho_s = np.outer(psi, psi)
z1 = np.array([1, 1, -1, -1]); z2 = np.array([1, -1, 1, -1])

print("p,shots,n00,n01,n10,n11,G_measured,G_exact,singlet_fraction_measured,singlet_fraction_exact,pair_entangled")
rows, counts = [], []
for p in a.p:
    rho = (1 - p) * rho_s + p * np.eye(4) / 4
    prob = np.real(np.diag(rho)); prob = prob / prob.sum()
    n = rng.multinomial(a.shots, prob)
    f = n / a.shots
    G = np.sum(f * z1 * z2) - np.sum(f * z1) * np.sum(f * z2)
    ps = 0.25 - 0.75 * G
    row = (p, a.shots, *n, G, -(1 - p), ps, 1 - 0.75 * p, int(ps > 0.5))
    rows.append(row); counts.append((p, *n))
    print(f"{p:.4f},{a.shots},{n[0]},{n[1]},{n[2]},{n[3]},{G:.5f},{-(1 - p):.5f},{ps:.5f},{1 - 0.75 * p:.5f},{int(ps > 0.5)}")
np.savetxt(os.path.join(a.out, "step8_two_qubits.csv"), rows, delimiter=",", fmt="%.6f", comments="",
           header="p,shots,n00,n01,n10,n11,G_measured,G_exact,singlet_fraction_measured,singlet_fraction_exact,pair_entangled")
np.savetxt(os.path.join(a.out, "step8_counts.csv"), counts, delimiter=",", fmt="%.6f", comments="",
           header="p,n00,n01,n10,n11")
