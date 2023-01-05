#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH -n 8
#SBATCH --gpus-per-task=4
#SBATCH --mem-per-gpu=20G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/8evzwxia" &
wait