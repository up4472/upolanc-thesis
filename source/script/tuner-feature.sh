#!/bin/bash

#SBATCH --job-name=cnn-raytune-feature
#SBATCH --output=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/raytune-%j.out
#SBATCH --error=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/raytune-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=6
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

# Default params
# --tuner_concurrent : 5          : [int] ...
# --tuner_trials     : 250        : [int] ...
# --tuner_grace      : 10         : [int] ...
# --model_epochs     : 50         : [int] ...
# --filter_id        : 0          : [int] ...
# --generator        : group      : [str] stratified, group, random

# Run script
python /d/hpc/home/up4472/workspace/upolanc-thesis/notebook/nbp06-tuner-feature.py \
--tuner_concurrent 5 \
--tuner_trials 1000 \
--tuner_grace 25 \
--model_epochs 50 \
--filter_id 2 \
--generator group
