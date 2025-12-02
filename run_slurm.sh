#!/bin/bash
#SBATCH --account=mixed_reality
#SBATCH --gpus=5060ti:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /etc/profile

module add cuda/12.8

nvidia-smi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies silently
#pip install torch torchvision > /dev/null
#pip install -r requirements.in > /dev/null

# Test GPU availability
# python test_cuda.py

python main.py opt:muon data:cifar10 arch:mlp --data.classes=4 --data.n=1000 --data.criterion=mse --opt.lr=0.02 --runs discrete --steps=2000 --eig.frequency=1
