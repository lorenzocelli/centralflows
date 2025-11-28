#!/bin/bash
#SBATCH --account=mixed_reality
#SBATCH --gpus=5060ti:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

module add cuda/12.8

nvidia-smi

source .venv/bin/activate

# Test GPU availability
# python test_cuda.py

python main.py data:cifar10 arch:mlp --data.classes=4 --data.n=1000 --data.criterion=mse --runs discrete --steps=2000 --eig.frequency=1
