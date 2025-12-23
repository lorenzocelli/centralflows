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

python main.py opt:adamw data:cifar10 arch:mlp --expid=adamw_fix_2 --data.classes=4 --data.n=1000 --data.criterion=mse --opt.lr=0.001 --runs discrete --steps=2000 --eig.frequency=1 --eig.track-threshold=None
# python main.py opt:rmsprop data:cifar10 arch:mlp --expid=rmsprop --data.classes=4 --data.n=1000 --data.criterion=mse --opt.lr=2e-5 --opt.beta2=0.99 --opt.eps=1e-7 --opt.bias-correction --runs discrete --steps=500 --eig.frequency=1 --warm-start 5 --eig.track-threshold 1.75