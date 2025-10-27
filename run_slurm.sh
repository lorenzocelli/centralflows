#!/bin/bash
#SBATCH --account=deep_learning
#SBATCH --gpus=5060ti:1
#SBATCH --output=logs/%x-%j.out

module load cuda/12.8

# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies silently
pip install torch torchvision > /dev/null
pip install -r requirements.in > /dev/null

# Test GPU availability
# python test_cuda.py

python main.py opt:gd data:cifar10 arch:mlp --data.classes=4 --data.n=1000 --data.criterion=mse --opt.lr=0.02 --runs discrete --steps=2000 --eig.frequency=1
