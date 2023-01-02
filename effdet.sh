#!/bin/bash

#SBATCH --job-name=EffDet
#SBATCH --time=30-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=6
#SBATCH --mem=100G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun PYTHONPATH="/home/users/mt361/COVID-Classification:/home/users/mt361/COVID-Classification/utils:$PYTHONPATH" python3 -u run.py -wdb True
wait
