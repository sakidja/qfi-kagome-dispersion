#!/bin/bash
# Runs the whole tutorial on the 12 site cluster. Start it inside the tutorial folder:
#     bash run_tutorial.sh 2>&1 | tee tutorial_out/log.txt      (after: mkdir -p tutorial_out)
# Every step can also be typed by hand; TUTORIAL.md explains each one.
set -e
S=../scripts; [ -d $S ] || S=..     # in the repository the scripts are in ../scripts
O=tutorial_out
mkdir -p $O
T="0.02 0.05 0.1 0.2 0.3"
# The matrices here are small. One thread per process is fastest; many threads on a shared node slow them down.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "===== step 1: two spins and three spins ====="
python step1_small_clusters.py --out $O

echo "===== step 2: clean 12 site kagome cluster ====="
python $S/qfi_kagome_ed.py --L1 2 --L2 2 --nvac 0 --T $T --out $O/run

echo "===== step 3: one vacancy, bonds sorted by distance from it ====="
python $S/qfi_kagome_ed.py --L1 2 --L2 2 --nvac 1 --nconf 6 --T $T --out $O/run
python $S/shell_test.py $O/run --tag L2x2 --nvac 1

echo "===== step 4: two and three vacancies, tables and averaged map ====="
for nv in 2 3; do
  python $S/qfi_kagome_ed.py --L1 2 --L2 2 --nvac $nv --nconf 6 --T $T --out $O/run
done
python $S/analyze_all.py $O/run --tag L2x2
python $S/plot_avg.py $O/run/analysis && mv avg_dose.png $O/

echo "===== step 5: dispersion maps at the lowest and the highest temperature ====="
python $S/plot_dispersion.py $O/run --tag L2x2 --Tindex 0
python $S/plot_dispersion.py $O/run --tag L2x2 --Tindex 4

echo "===== step 6: local picture and its melting ====="
python $S/local_map.py $O/run --tag L2x2
python $S/melt_movie.py $O/run --tag L2x2 --nvac 1

echo "===== step 7: twisted boundary conditions, finer wave vector grid ====="
python $S/qfi_kagome_twist.py --L1 2 --L2 2 --nvac 1 --nconf 2 --nt 2 2 --T $T --out $O/twist
python $S/plot_twist.py $O/twist --tag L2x2 --line all

echo "===== step 8: two qubits ====="
python step8_two_qubits.py --out $O

echo "===== step 9: QFI matrix between wave vectors, size and phase ====="
python $S/plot_qq.py $O/run --tag L2x2
python $S/plot_qq.py $O/run --tag L2x2 --phase

echo "===== files written ====="
find $O -type f | sort
