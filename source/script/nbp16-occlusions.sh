#!/bin/bash

#SBATCH --job-name=cnn-occlusion
#SBATCH --output=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/occlusion-%j.out
#SBATCH --error=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/occlusion-%j.err
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
# --model_folder     : ...      : [str] ...
# --occlusion_size   : 10       : [int] ...
# --occlusion_stride : 10       : [int] ...
# --occlusion_type   : zero     : [int] zero, shuffle, random
# --occlusion_method : window   : [str] window, region
# --relevance_type   : r2       : [str] r2, mse, mae

# Run script
python /d/hpc/projects/FRI/up4472/upolanc-thesis/notebook/nbp16-occlusion.py \
--model_folder washburn-0-tf2150-f2-0250-77-tissue-mean-explode \
--occlusion_size 10 \
--occlusion_stride 1 \
--occlusion_type zero \
--occlusion_method window \
--relevance_type r2
