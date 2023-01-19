#!/bin/bash
#SBATCH --job-name=EffDetD0
#SBATCH --time=90-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-gpu=10G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate covid-cv

srun python3 run.py -p 'config/efficientdet_d0.yaml' -mode 'train'
wait
