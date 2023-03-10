#!/bin/bash

#SBATCH --job-name=cnn-raytune-washburn
#SBATCH --output=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/raytune-%j.out
#SBATCH --error=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/raytune-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --time=4-00:00:00

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
python /d/hpc/home/up4472/workspace/upolanc-thesis/notebook/nbp06-tuner.py \
--target_group global \
--target_type mean \
--target_explode false \
--target_filter none \
--model_name washburn2019 \
--model_epochs 25 \
--tuner_concurrent 5 \
--tuner_trials 500 \
--tuner_grace 10
