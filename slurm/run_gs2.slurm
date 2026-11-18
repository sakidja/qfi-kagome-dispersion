#!/bin/bash
#SBATCH -A YOUR_ACCOUNT
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -J kag_gs2
#SBATCH --array=0-5
#SBATCH -o gs2_%A_%a.out
# 27 site kagome (3x3) ground state with TWO vacancies at fixed concentration,
# placed at increasing separation. Graph distance d = number of bonds between the holes.
#   task 0: sites 0 1    d = 1  adjacent (same triangle)
#   task 1: sites 0 7    d = 2  bent     (straight line distance 1.73)
#   task 2: sites 0 3    d = 2  in line  (straight line distance 2)
#   task 3: sites 0 4    d = 3
#   task 4: sites 0 12   d = 4  the largest separation the cluster allows
#   task 5: site  0      one hole again, now with the degeneracy check
module load python
cd $SLURM_SUBMIT_DIR
CASES=("0 1" "0 7" "0 3" "0 4" "0 12" "0")
python kagome_gs.py --L1 3 --L2 3 --vac ${CASES[$SLURM_ARRAY_TASK_ID]} --out gs2
