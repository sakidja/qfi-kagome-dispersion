#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH -J melt_disp
#SBATCH -o melt_disp_%j.out
# Melting of the dispersion map: 12 site kagome cluster, clean and with one vacancy,
# ten temperatures from 0.02 J to 2 J, 4 x 4 twists (256 wave vectors), WITH spectra.
# One process per twist, all 32 at the same time on one node. Output folder: tmelt_tw
module load python
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
cd $SLURM_SUBMIT_DIR
T="0.02 0.05 0.1 0.2 0.3 0.5 0.7 1.0 1.5 2.0"
mkdir -p tmelt_tw
for t1 in 0 1 2 3; do
  for t2 in 0 1 2 3; do
    python qfi_kagome_twist.py --L1 2 --L2 2 --nvac 0 --nconf 1 --seed 1234 --nt 4 4 --twist $t1 $t2 \
        --T $T --eps 1e-8 --out tmelt_tw > tmelt_tw/log_nv0_t${t1}_${t2}.txt 2>&1 &
    python qfi_kagome_twist.py --L1 2 --L2 2 --nvac 1 --nconf 6 --seed 1234 --nt 4 4 --twist $t1 $t2 \
        --T $T --eps 1e-8 --out tmelt_tw > tmelt_tw/log_nv1_t${t1}_${t2}.txt 2>&1 &
  done
done
wait
echo "files written: $(ls tmelt_tw/*.npz | wc -l)  (expected 112)"
grep -h "check F <= G" tmelt_tw/log_nv*.txt | sort | uniq -c | head -n 5
