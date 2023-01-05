#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH -n 15
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=50G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &

srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &

srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
srun --exclusive -n1 wandb agent "muhang-tian/EfficientDet D0 Sweep/0tzeqidi" &
wait