#!/bin/bash
#SBATCH --account=mixed_reality
#SBATCH --gpus=5060ti:1
#SBATCH --output=logs/%x-%j.out

module add cuda/12.8

# Set up virtual environment
# python3 -m venv .venv
source .venv/bin/activate

# Install dependencies silently
# pip install torch torchvision > /dev/null
# pip install -r requirements.in > /dev/null

# Test GPU availability
# python test_cuda.py

python main.py opt:comp data:regression arch:linear \
        --opt.lr=0.1 --data.n=1000 --runs discrete \
        --steps=1000 --eig.frequency=1 --warm-start 5 \
        --expid="linear" --eig.track-threshold None \
        --opt.frozen-muon

# Convergence with Muon + linear model
# Note: setting a very high eig.track-threshold ensures we track only the max eigenvalue
# python main.py opt:comp data:regression arch:linear --opt.lr=0.1 --data.n=1000 --runs discrete --steps=1000 --eig.frequency=1 --warm-start 5 --eig.track-threshold 10000000

# Convergence with GD + linear model
# python main.py opt:gd data:regression arch:linear --opt.lr=0.000001958 --data.n=10000 --data.criterion=mse --runs discrete --steps=2000 --eig.frequency=1 --warm-start 5 --eig.track-threshold 1.75
