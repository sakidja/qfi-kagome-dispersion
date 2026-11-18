#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 02:00:00
#SBATCH -J kag_gs
#SBATCH --array=0-3
#SBATCH -o gs_%A_%a.out
# 27 site kagome (3x3) ground state. Task 0: clean. Tasks 1 to 3: one hole on site 0, 1, 2
# (one per sublattice; they should agree if the cluster treats the sublattices alike).
module load python
cd $SLURM_SUBMIT_DIR
L1=${L1:-3}; L2=${L2:-3}
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python kagome_gs.py --L1 $L1 --L2 $L2 --out gs
else
  python kagome_gs.py --L1 $L1 --L2 $L2 --vac $((SLURM_ARRAY_TASK_ID-1)) --out gs
fi
