#!/bin/bash

#SBATCH --job-name=bert-dnabert
#SBATCH --output=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/dnabert-%j.out
#SBATCH --error=/d/hpc/home/up4472/workspace/upolanc-thesis/slurm/dnabert-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=64G
#SBATCH --cpus-per-gpu=12
#SBATCH --time=2-00:00:00

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
export KMER=6
export TARGET=global-mean

export MODEL_PATH=/d/hpc/home/up4472/workspace/upolanc-thesis/resources/dnabert/$KMER-new-12w-0
export DATA_PATH=/d/hpc/home/up4472/workspace/upolanc-thesis/output/nbp05-target/dnabert-$KMER-prom/$TARGET
export OUTPUT_PATH=/d/hpc/home/up4472/workspace/upolanc-thesis/output/nbp10-dnabert/$KMER

python /d/hpc/home/up4472/workspace/upolanc-thesis/notebook/nbp10-dnabert.py \
--model_type rbertfc3 \
--tokenizer_name=dna$KMER \
--model_name_or_path $MODEL_PATH \
--task_name regression \
--do_train \
--do_eval \
--data_dir $DATA_PATH \
--max_seq_length 512 \
--per_gpu_eval_batch_size=32 \
--per_gpu_train_batch_size=32 \
--learning_rate 5e-5 \
--num_train_epochs 25.0 \
--output_dir $OUTPUT_PATH \
--evaluate_during_training \
--logging_steps 100 \
--save_steps 4000 \
--warmup_percent 0.1 \
--hidden_dropout_prob 0.1 \
--overwrite_output \
--weight_decay 0.01 \
--n_process 12
