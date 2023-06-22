#!/bin/bash

#SBATCH --job-name=cnn-bert
#SBATCH --output=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/bert-%j.out
#SBATCH --error=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/bert-%j.err
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
# --bert_arch      : febert         : [str] febert
# --bert_kmer      : 3              : [str] 3, 6
# --bert_sequence  : promoter-512   : [str] promoter-512, promoter-4096, promoter-utr5-4096, transcript-2150
# --bert_target    : global-mean    : [str] global-mean, tissue-mean-explode, group-mean-explode
# --model_epochs   : 50             : [int] ...
# --model_params   : none           : [int] none
# --model_mode     : regression     : [str] regression, classification
# --model_arch     : fc2            : [str] fc2, fc3
# --filter_id      : 0              : [int] ...

# Run script
python /d/hpc/projects/FRI/up4472/upolanc-thesis/notebook/nbp12-bert-cnn.py \
--bert_arch febert \
--bert_kmer 6 \
--bert_sequence promoter-512 \
--bert_target global-mean \
--model_epochs 250 \
--model_params 0 \
--model_mode regression \
--model_arch fc3 \
--filter_id 3 \
