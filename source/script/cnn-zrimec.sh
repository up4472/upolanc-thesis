#!/bin/bash

#SBATCH --job-name=cnn-zrimec
#SBATCH --output=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/zrimec-%j.out
#SBATCH --error=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/zrimec-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=6
#SBATCH --time=1-00:00:00

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
# --target_group   : global         : [str] tissue, group, age, global, perturbation
# --target_type    : mean           : [str] mean, max
# --target_explode : false          : [str] true, false
# --target_filter  : none           : [str] none
# --model_epochs   : 250            : [int] ...
# --model_params   : none           : [int] none
# --model_mode     : regression     : [str] regression, classification
# --filter_id      : 0              : [int] ...
# --generator      : group          : [str] stratified, group, random
# --features       : true           : [str] true, false

# Run script
python /d/hpc/home/up4472/workspace/upolanc-thesis/notebook/nbp07-zrimec.py \
--target_group global \
--target_type mean \
--target_explode false \
--target_filter none \
--model_epochs 500 \
--model_params 0 \
--model_mode regression \
--filter_id 0 \
--generator group \
--features true

