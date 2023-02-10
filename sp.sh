#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH --array=1-30
#SBATCH -n 1
#SBATCH --cpus-per-task=3
#SBATCH -p compsci
#SBATCH --mem-per-cpu=30G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate covid-cv

sleep $((SLURM_ARRAY_TASK_ID*10))
srun wandb agent --count 1 "muhang-tian/EfficientDetD0/w5rud6ym"