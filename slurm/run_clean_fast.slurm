#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 10:00:00
#SBATCH -J qfi_clean
#SBATCH -o qfi_clean_%j.out
# Clean 18 site kagome reference (no vacancy), fast path:
#   only the populated eigenvectors, sectors nup <= 9 (the others follow by spin flip symmetry),
#   and a checkpoint after every sector. If the job stops for any reason, submit this same
#   script again: it continues from the last finished sector.
module load python
export OMP_NUM_THREADS=128
export OPENBLAS_NUM_THREADS=128
export MKL_NUM_THREADS=128
cd $SLURM_SUBMIT_DIR
python qfi_kagome_ed.py --L1 2 --L2 3 --nvac 0 \
    --T 0.02 0.05 0.1 0.2 0.3 --eps 1e-8 --mem_gb 60 --no_spectra --fast --out results_clean
