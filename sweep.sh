#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=60-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=8
#SBATCH --mem=100G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun wandb agent --count 1 "muhang-tian/EfficientDetV2 DS Sweep/f23ux2xu"
wait