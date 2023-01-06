#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH --array=1-20
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=30G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate covid-cv

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID 
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

srun wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/m77pgogj"