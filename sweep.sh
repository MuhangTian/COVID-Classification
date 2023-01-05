#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH -N 5
#SBATCH -n 5
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-gpu=20G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun --exclusive -n1 -N1 wandb agent --count 1 "muhang-tian/EfficientDet D0 Sweep/jzdyll97" &
srun --exclusive -n1 -N1 wandb agent --count 1 "muhang-tian/EfficientDet D0 Sweep/jzdyll97" &
srun --exclusive -n1 -N1 wandb agent --count 1 "muhang-tian/EfficientDet D0 Sweep/jzdyll97" &
srun --exclusive -n1 -N1 wandb agent --count 1 "muhang-tian/EfficientDet D0 Sweep/jzdyll97" &
srun --exclusive -n1 -N1 wandb agent --count 1 "muhang-tian/EfficientDet D0 Sweep/jzdyll97" &