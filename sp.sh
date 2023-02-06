#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH --array=1-30
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=30G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate covid-cv

srun wandb agent --count 1 "muhang-tian/EfficientDetD0 Sweep (Random)/2cqidr0t"
wait