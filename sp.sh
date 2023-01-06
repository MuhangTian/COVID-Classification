#!/bin/bash
#SBATCH --job-name=SP-EffDet
#SBATCH --time=90-00:00:00
#SBATCH --nodes=20
#SBATCH -n 20
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=40G
#SBATCH --mail-user=muhang.tian@duke.edu
#SBATCH --output=None
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &

srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &

srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &

srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
srun --exclusive --nodes=1 -n1 wandb agent "muhang-tian/EfficientDetD0 Sweep (Random)/fkjepv2n" &
wait