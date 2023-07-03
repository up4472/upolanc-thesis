#!/bin/bash

#SBATCH --job-name=cnn-pipeline
#SBATCH --output=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/pipeline-%j.out
#SBATCH --error=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/pipeline-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=6
#SBATCH --time=4-00:00:00

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
# --model_arch     : zrimec          : [str] zrimec, washburn
# --model_epochs   : 250             : [int] ...
# --model_params   : none            : [int] none
# --filter_id      : 0               : [int] ...
# --generator      : group           : [str] stratified, group, random
# --features       : true            : [str] true, false
# --sequence_start : none            : [int] none
# --sequence_end   : none            : [int] none
# --sequence_type  : transcript-2150 : [str] transcript-2150, transcript-6150, promoter-full-5000, promoter-utr5-5000

# Run script
python /d/hpc/projects/FRI/up4472/upolanc-thesis/notebook/nbp14-pipeline.py \
--target_group global \
--target_type mean \
--target_explode false \
--target_filter none \
--model_arch zrimec \
--model_epochs 500 \
--model_params 0 \
--filter_id 3 \
--features true \
--sequence_start none \
--sequence_end none \
--sequence_type transcript-2150
