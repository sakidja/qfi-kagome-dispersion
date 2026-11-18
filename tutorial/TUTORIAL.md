# Tutorial: the QFI dispersion map, step by step

This tutorial follows the paper from the two spin problem to the diluted kagome cluster. Everything runs on the 12 site cluster. The matrices are small, so use one thread per process (`run_tutorial.sh` sets this); with many threads on a shared node the same runs are far slower. Each step matches an exercise of Section 6 of the paper; the answers are in Section 7.

All commands are typed inside the `tutorial` folder. To run every step at once:

```
mkdir -p tutorial_out
bash run_tutorial.sh 2>&1 | tee tutorial_out/log.txt
```

The folder `tutorial_out` in this repository holds the output of that command from the author's run, so you can compare your numbers with it line by line.

## The quantities

* **G** for a bond: 4 times the connected correlation of Sz on its two spins. A perfect singlet has G = −1. No correlation gives G = 0.
* **Singlet fraction** p_s = 1/4 − 3G/4. If p_s is above 1/2, the two spins are entangled as a pair.
* **g(q)**: the equal time fluctuation at wave vector q (the structure factor, from G).
* **f(q)**: the quantum Fisher information density at wave vector q. It never exceeds g(q). The part of g that f does not contain is thermal. If f(q) is above 1, the state is entangled.
* **Quantum share**: f divided by g.

## Step 1. Two spins and three spins (exercises 1 and 2)

```
python step1_small_clusters.py --out tutorial_out
```

What to look for:

* Dimer: the column `ratio` equals the column `tanh2(J/2T)` at every temperature. At low temperature G = −1 and f per site = 2.
* Triangle: G = −1/3 and singlet fraction 1/2 at low temperature. Three spins on a triangle cannot all pair, so no bond reaches the singlet value.
* Remove one corner of the triangle and you are back at the dimer: G goes from −1/3 to −1. This is the whole vacancy effect in its smallest form.

## Step 2. The clean kagome cluster

```
python ../scripts/qfi_kagome_ed.py --L1 2 --L2 2 --nvac 0 --T 0.02 0.05 0.1 0.2 0.3 --out tutorial_out/run
```

The script builds the 12 site cluster with periodic boundaries, diagonalizes the Hamiltonian in each sector of total Sz, and writes the matrices F and G for every temperature. It prints two checks: f never exceeds g, and f/g at the lowest temperature.

## Step 3. One vacancy and the freed bond (exercise 3)

```
python ../scripts/qfi_kagome_ed.py --L1 2 --L2 2 --nvac 1 --nconf 6 --T 0.02 0.05 0.1 0.2 0.3 --out tutorial_out/run
python ../scripts/shell_test.py tutorial_out/run --tag L2x2 --nvac 1
```

The second script sorts the bonds by the distance of their two ends from the vacancy. The label (1,1) is the bond between the two neighbors of the vacancy that shared a triangle with it.

What to look for:

* The (1,1) bond is strongly negative, with singlet fraction above 1/2: the two spins freed by the vacancy lock into a pair.
* The (1,2) bonds next to it are weak.
* On heating, the outer bonds lose their pattern first and the (1,1) bond last.

## Step 4. More vacancies and the averaged map (exercises 5 and 6)

```
python ../scripts/qfi_kagome_ed.py --L1 2 --L2 2 --nvac 2 --nconf 6 --T 0.02 0.05 0.1 0.2 0.3 --out tutorial_out/run
python ../scripts/qfi_kagome_ed.py --L1 2 --L2 2 --nvac 3 --nconf 6 --T 0.02 0.05 0.1 0.2 0.3 --out tutorial_out/run
python ../scripts/analyze_all.py tutorial_out/run --tag L2x2
python ../scripts/plot_avg.py tutorial_out/run/analysis
```

What to look for in the printed tables:

* `averaged.csv`: compare `peak_of_avg_map` with `mean_of_config_peaks`. The second is never smaller. Each configuration peaks at its own wave vector; averaging the maps first restores the lattice symmetry (exercise 5).
* `cross_q.csv`: the cross q share is exactly zero without vacancies and grows with their number. It is lower at T = 0.3 than at T = 0.02 (exercise 6).
* `witness_T.csv`: the temperature at which the peak of f falls to 1, for the averaged map and for the mean of the per configuration peaks. The two differ, for the reason of exercise 5.
* Even and odd spin count: with an odd number of spins one spin is unpaired. The run prints `min_q f/g = 0.0000` for these cases: at q = 0 the structure factor is g = 1/N and the QFI is f = 0 (exercise 4).

## Step 5. The dispersion picture

```
python ../scripts/plot_dispersion.py tutorial_out/run --tag L2x2 --Tindex 0
python ../scripts/plot_dispersion.py tutorial_out/run --tag L2x2 --Tindex 4
```

Each figure shows f(Q) and g(Q), the structure factor S(Q, ω), and the QFI density in wave vector and energy. Read it like a phonon dispersion: low energy weight is the collective part, flat levels near J are pair excitations, and the tanh factor removes the low energy thermal weight first when the temperature rises. Compare the two temperatures.

## Step 6. The local picture and its melting

```
python ../scripts/local_map.py tutorial_out/run --tag L2x2
python ../scripts/melt_movie.py tutorial_out/run --tag L2x2 --nvac 1
```

The first draws, bond by bond, the share of the fluctuation that is not quantum. The second draws the bond map at each temperature, as single frames, as a strip and as an animated GIF.

## Step 7. Twisted boundaries (exercise 7)

```
python ../scripts/qfi_kagome_twist.py --L1 2 --L2 2 --nvac 1 --nconf 2 --nt 2 2 --T 0.02 0.05 0.1 0.2 0.3 --out tutorial_out/twist
python ../scripts/plot_twist.py tutorial_out/twist --tag L2x2 --line all
```

A 12 site cluster has few wave vectors. A twist of the boundary condition shifts them, and a 2 by 2 grid of twists gives four times as many. A twist of one full step returns the same spectrum with every wave vector moved by one step.

## Step 8. Two qubits (exercise 8)

```
python step8_two_qubits.py --out tutorial_out
```

The circuit X, X, H, CNOT prepares the singlet. Measuring both qubits in the Z basis gives G. With a depolarizing error of strength p, G = −(1 − p) and the singlet fraction is 1 − 3p/4; the pair stays entangled for p below 2/3. The script prints the measured and the exact values and saves the raw counts. The same circuit runs on any quantum simulator or processor.

## Going further

* 18 sites: the same commands with `--L1 2 --L2 3`. This needs a compute node; the job scripts are in `slurm/`.
* 27 sites, ground state only: `python ../scripts/kagome_gs.py --L1 3 --L2 3 --vac 0 --out gs` and `python ../scripts/gs_map.py gs`.
