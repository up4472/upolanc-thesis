#!/bin/bash

#SBATCH --job-name=up4472-zrimec2020r
#SBATCH --output=/d/hpc/home/up4472/workspace/upolanc-thesis/log/zrimec2020r-stdout.txt
#SBATCH --error=/d/hpc/home/up4472/workspace/upolanc-thesis/log/zrimec2020r-stderr.txt
#SBATCH --nodes=6
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --time=0-04:00:00

source activate /d/hpc/home/up4472/anaconda3

python /d/hpc/home/up4472/workspace/upolanc-thesis/lab/nbp07-zrimec2020r.py
