#!/bin/bash

#SBATCH --job-name=up4472-tuner
#SBATCH --output=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/tuner-stdout.txt
#SBATCH --error=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/tuner-stderr.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=6
#SBATCH --time=1-12:00:00

# Activate conda enviorment
source activate /d/hpc/home/up4472/anaconda3

# Define root path
ROOT="/d/hpc/home/up4472/workspace/upolanc-thesis/"

# Ensure PYTHONPATH contains root
if [[ ":$PYTHONPATH:" != *":$ROOT:"* ]]; then
	export PYTHONPATH="$PYTHONPATH:$ROOT"
fi

# Ensure PATH contains root
if [[ ":$PATH:" != *":$ROOT:"* ]]; then
	export PATH="$PATH:$ROOT"
fi

# Run script
python /d/hpc/home/up4472/workspace/upolanc-thesis/notebook/nbp06-tuner.py
