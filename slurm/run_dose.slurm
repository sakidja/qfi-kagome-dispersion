#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 03:00:00
#SBATCH -J qfi_dose
#SBATCH -o qfi_dose_%A_%a.out
# One vacancy configuration per array task. Submit once per dose, e.g.
#   sbatch --array=0-5 --export=ALL,NVAC=1,NCONF=6 run_dose.slurm
#   sbatch --array=0-7 --export=ALL,NVAC=2,NCONF=8 run_dose.slurm
#   sbatch --array=0-7 --export=ALL,NVAC=3,NCONF=8 run_dose.slurm
# NCONF must be the same in every task of a dose: the vacancy sets are drawn
# from a fixed seed, and --conf picks one of them.
module load python
export OMP_NUM_THREADS=128
export OPENBLAS_NUM_THREADS=128
export MKL_NUM_THREADS=128
cd $SLURM_SUBMIT_DIR
python qfi_kagome_ed.py --L1 2 --L2 3 --nvac ${NVAC} --nconf ${NCONF} \
    --conf ${SLURM_ARRAY_TASK_ID} --seed 1234 \
    --T 0.02 0.05 0.1 0.2 0.3 --eps 1e-8 --mem_gb 40 --out results
