#!/bin/bash

#SBATCH --job-name=cnn-dense
#SBATCH --output=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/dense-%j.out
#SBATCH --error=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/dense-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=32G
#SBATCH --cpus-per-gpu=6
#SBATCH --time=1-00:00:00

# Activate conda enviorment
source activate /d/hpc/projects/FRI/up4472/anaconda3

# Define root path
ROOT="/d/hpc/projects/FRI/up4472/upolanc-thesis/"

# Ensure PYTHONPATH contains root
if [[ ":$PYTHONPATH:" != *":$ROOT:"* ]]; then
	export PYTHONPATH="$PYTHONPATH:$ROOT"
fi

# Ensure PATH contains root
if [[ ":$PATH:" != *":$ROOT:"* ]]; then
	export PATH="$PATH:$ROOT"
fi

# Default params
# --target_group   : global          : [str] tissue, group, age, global, perturbation
# --target_type    : mean            : [str] mean, max
# --target_explode : false           : [str] true, false
# --target_filter  : none            : [str] none
# --model_epochs   : 50              : [int] ...
# --model_mode     : regression      : [str] regression, classification
# --model_arch     : fc3             : [str] fc2, fc3
# --model_type     : zrimec          : [str] zrimec, washburn
# --model_params   : none            : [int] none
# --filter_id      : 2               : [int] ...

# Run script
python /d/hpc/projects/FRI/up4472/upolanc-thesis/notebook/nbp10-dense.py \
--target_group global \
--target_type mean \
--target_explode false \
--target_filter none \
--model_epochs 250 \
--model_mode regression \
--model_arch fc3 \
--model_type zrimec \
--model_params 0 \
--filter_id 2
