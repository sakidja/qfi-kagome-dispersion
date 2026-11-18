#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 08:00:00
#SBATCH -J qfi_clean
#SBATCH -o qfi_clean_%j.out
# Clean 18 site kagome reference (dose 0). Largest Sz sector D = 48620:
# dense H ~19 GB, eigh workspace ~2x more; a 512 GB CPU node is comfortable.
module load python
export OMP_NUM_THREADS=128
export OPENBLAS_NUM_THREADS=128
export MKL_NUM_THREADS=128
cd $SLURM_SUBMIT_DIR
python qfi_kagome_ed.py --L1 2 --L2 3 --nvac 0 \
    --T 0.02 0.05 0.1 0.2 0.3 --eps 1e-8 --mem_gb 40 --out results
