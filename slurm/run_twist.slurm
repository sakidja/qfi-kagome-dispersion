#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 06:00:00
#SBATCH -J qfi_twist
#SBATCH -o qfi_twist_%A_%a.out
# One twist per array task (4 x 4 grid = tasks 0 to 15); each task runs all
# NCONF vacancy configurations of the dose. Example:
#   sbatch --array=0-15 --export=ALL,NVAC=3,NCONF=4 run_twist.slurm
module load python
export OMP_NUM_THREADS=128
export OPENBLAS_NUM_THREADS=128
export MKL_NUM_THREADS=128
cd $SLURM_SUBMIT_DIR
T1=$(( SLURM_ARRAY_TASK_ID / 4 ))
T2=$(( SLURM_ARRAY_TASK_ID % 4 ))
python qfi_kagome_twist.py --L1 2 --L2 3 --nvac ${NVAC} --nconf ${NCONF} --seed 1234 \
    --nt 4 4 --twist ${T1} ${T2} --T 0.02 0.05 0.1 0.2 0.3 --eps 1e-8 --mem_gb 60 \
    --out results_twist
