#!/bin/bash

#SBATCH --job-name=EffDet
#SBATCH --time=30-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=5
#SBATCH --mem=50G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun python3 -u run.py -wdb True
wait
