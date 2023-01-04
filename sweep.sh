#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=30-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=4
#SBATCH --mem=100G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=sweep.out
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun wandb agent "muhang-tian/EfficientDetV2 DS Sweep/f23ux2xu"
wait