#!/bin/bash

#SBATCH --job-name=cnn-raytune-zrimec
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

# Default params
# --target_group     : global       : [str] tissue, group, age, global, perturbation
# --target_type      : mean         : [str] mean, max
# --target_explode   : false        : [str] true, false
# --target_filter    : none         : [str] none
# --model_name       : zrimec2020   : [str] zrimec2020, washburn2019
# --model_type       : r            : [str] r, c
# --model_epochs     : 500          : [int] ...
# --tuner_concurrent : 5            : [int] ...
# --tuner_trials     : 250          : [int] ...
# --tuner_grace      : 10           : [int] ...
# --param_share      : false        : [str] true, false
# --filter_id        : 0            : [int] ...
# --generator        : group        : [str] stratified, group, random
# --features         : true         : [str] true, false

# Run script
python /d/hpc/home/up4472/workspace/upolanc-thesis/notebook/nbp06-tuner-model.py \
--target_group global \
--target_type mean \
--target_explode false \
--target_filter none \
--model_name zrimec2020 \
--model_type r \
--model_epochs 25 \
--tuner_concurrent 5 \
--tuner_trials 1000 \
--tuner_grace 10 \
--param_share false \
--filter_id 0 \
--generator group \
--features true
