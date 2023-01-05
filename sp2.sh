#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-gpu=20G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

CUDA_VISIBLE_DEVICES=2 wandb agent --count 1 "muhang-tian/EfficientDet D0 Sweep/jzdyll97" 