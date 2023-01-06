#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=3
#SBATCH --mem-per-gpu=10G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n"