# QFI dispersion map on the diluted kagome lattice

Code and data for the paper

**Quantum Fisher information as a dispersion map for magnetic states: a worked example on the diluted kagome lattice**

Ridwan Sakidja, Department of Physics, Astronomy and Materials Science, Missouri State University

Preprint: Zenodo, https://doi.org/10.5281/zenodo.24331367

Repository: https://github.com/sakidja/qfi-kagome-dispersion

The model is the spin 1/2 Heisenberg antiferromagnet on the kagome lattice with site vacancies, on periodic clusters of 12, 18 and 27 sites. The scripts compute the quantum Fisher information (QFI) matrix of the thermal state by exact diagonalization, map it in wave vector and energy, and resolve it bond by bond around the vacancies.

## Quick start

```
git clone https://github.com/sakidja/qfi-kagome-dispersion.git
cd qfi-kagome-dispersion/tutorial
mkdir -p tutorial_out
bash run_tutorial.sh 2>&1 | tee tutorial_out/log.txt
```

This runs the 12 site example of the paper from start to end. The steps are explained in `tutorial/TUTORIAL.md`.

## Requirements

Python 3 with numpy, scipy and matplotlib. Pillow is needed only for the animated GIF of melt_movie.py.

The 12 site runs are small and fit on a laptop; use one thread per process. The 18 site runs need a compute node (the clean 18 site reference took about 7 hours on one CPU node with the fast path). The 27 site runs are ground state only (Lanczos).

## Folders

* `scripts/` the Python scripts
* `slurm/` the job scripts as used; set your own account in the line `#SBATCH -A YOUR_ACCOUNT`
* `results/` the result files of the runs (npz per configuration, csv tables, run logs)
* `figures/` the figures of the paper as produced by the scripts
* `tutorial/` a step by step tutorial on the 12 site cluster, with its own output for comparison

## Scripts

| Script | What it does |
|---|---|
| `qfi_kagome_ed.py` | Thermal state by full exact diagonalization; QFI matrix F, equal time matrix G, f(q), g(q), spectra. Option `--fast` keeps only the populated eigenvectors, uses spin flip symmetry and saves a checkpoint after every sector. |
| `qfi_kagome_twist.py` | The same with twisted boundary conditions, to refine the wave vector grid. |
| `kagome_gs.py` | Lanczos ground state for larger clusters, bond correlations by distance from the vacancy, singlet fraction, degeneracy check. |
| `analyze_all.py` | Tables from the finite temperature runs: same q, zones, cross q share, witness temperature, averaged map, energy centroid. |
| `bond_test.py` | Bond correlation G by class (freed, near, far). |
| `shell_test.py` | Bond correlation, quantum share and singlet fraction by graph distance from the vacancy. |
| `plot_avg.py` | Averaged map against vacancy number. |
| `plot_twist.py`, `plot_dispersion.py` | Dispersion maps: f(Q), g(Q), S(Q, ω) and QFI density. |
| `plot_qq.py` | Heat map of the QFI matrix in the wave vector basis, q against q′: block diagonal for the clean lattice, weight outside the blocks with vacancies. With `--phase`: single configurations (clean, one vacancy at two positions) with the magnitude and the phase of each element. |
| `local_map.py` | Local coherence map on the lattice. |
| `gs_map.py` | Bond maps of the ground state runs. |
| `melt_movie.py` | Bond map against temperature, as frames, strip and GIF. |
| `melt_dispersion.py` | Dispersion maps S(Q, ω) and QFI density at several temperatures side by side, with the g and f densities against energy; writes the numbers behind the figure. |
| `draw_circuit_fig.py` | Schematic of the lattice as a quantum circuit: each site a qubit wire, each bond a partial SWAP gate. No computed data. |

## Tables and figures of the paper

The commands are given as run from one working folder that holds the scripts and the result folders. In this repository the scripts are in `scripts/` and the result folders are under `results/`; copy the scripts next to the result folders, or give the paths.

| Item | Command |
|---|---|
| Figure 1 | `python draw_circuit_fig.py` |
| Figure 2 | `python plot_twist.py results_twist --tag L2x3 --line all --paper` |
| Table 2 | `python analyze_all.py results --tag L2x3` |
| Figure 3 | `python plot_avg.py results/analysis --paper` |
| Figure 4 | `python plot_qq.py tw12 --tag L2x2 --paper` |
| Figure 5 | `python plot_qq.py results --tag L2x3 --phase --paper` |
| Figure 6 | `python plot_twist.py tw12 --tag L2x2 --line all --paper` |
| Table 4 | `python bond_test.py results --tag L2x3 --E0clean -8.0482707735 --nbonds 36` |
| Figure 7 | `python local_map.py results --tag L2x3 --paper` |
| Table 5 | `python shell_test.py results --tag L2x3 --nvac 1` |
| Table 6, Figure 8 | `sbatch run_gs.slurm`, then `python gs_map.py gs` |
| Table 7, Figure 9 | `python qfi_kagome_ed.py --L1 2 --L2 2 --nvac 1 --nconf 6 --T 0.02 0.05 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0 --no_spectra --out tmelt`, then `python shell_test.py tmelt --tag L2x2 --nvac 1` and `python melt_movie.py tmelt --tag L2x2 --nvac 1` |
| Table 8, Figure 10 | `sbatch run_melt_disp.slurm`, then `python melt_dispersion.py tmelt_tw --tag L2x2 --nvac 1 --paper` and the same with `--nvac 0` (tables `melt_weights_*.csv`) |
| Table 9, Figure 11 | `sbatch run_gs2.slurm`, then `python gs_map.py gs2` |
| Clean 18 site column of Table 2 | `sbatch run_clean_fast.slurm`, then copy `results_clean/L2x3_nv0_c0.npz` into `results/` |

The production runs behind `results/`, `results_twist/` and `tw12/` are given by the job scripts in `slurm/`.

## Tutorial

The folder `tutorial/` runs the paper step by step on the 12 site cluster, from two spins by hand to a two qubit measurement. Start with `tutorial/TUTORIAL.md`.

```
cd tutorial
mkdir -p tutorial_out
bash run_tutorial.sh 2>&1 | tee tutorial_out/log.txt
```

## How to cite

R. Sakidja, "Quantum Fisher information as a dispersion map for magnetic states: a worked example on the diluted kagome lattice", preprint, Zenodo (2026), https://doi.org/10.5281/zenodo.24331367. Code and data: https://github.com/sakidja/qfi-kagome-dispersion

## License

The code is released under the MIT License (file `LICENSE`).

## Acknowledgment

This research used resources of the National Energy Research Scientific Computing Center (NERSC), a Department of Energy Office of Science User Facility. AI (LLM) assistance (Claude, Anthropic) was used in preparing code and text. The framework, calculations and analysis are the author's, who takes full responsibility for the scientific content.
