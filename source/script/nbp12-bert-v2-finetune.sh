#!/bin/bash

#SBATCH --job-name=bert-v2-finetune
#SBATCH --output=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/bert-v2-finetune-%j.out
#SBATCH --error=/d/hpc/projects/FRI/up4472/upolanc-thesis/slurm/bert-v2-finetune-%j.err
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

# TARGET   : global-mean, tissue-mean-explode, tissue-mean-seedling, group-mean-explode
# SEQUENCE : promoter-512, promoter-4096, promoter-utr5-4096, transcript-2150
# FILTER   : filter1, filter2, filter3, filter4, filter5, filter6
# MODELS   : febert_v2, rbertfc1_v2, rbertfc3_v2, rbertfc3_def_v2, rbertfc3_rnn_v2, rbertfc3_cat_v2

# Run script
export DATA_KMER=3
export DATA_TARGET=global-mean
export DATA_SEQUENCE=promoter-512
export DATA_FILTER=filter2
export NAME_MODEL=rbertfc3_def_v2

export PATH_ROOT=/d/hpc/projects/FRI/up4472/upolanc-thesis
export PATH_BERT=$PATH_ROOT/resources/dnabert/bpe-new-12w-0
export PATH_DATA=$PATH_ROOT/output/nbp05-target/$DATA_FILTER/dnabert-6/$DATA_SEQUENCE/$DATA_TARGET
export PATH_OUTS=$PATH_ROOT/output/nbp12-bert/bert/$DATA_FILTER/$NAME_MODEL/bpe/$DATA_SEQUENCE/$DATA_TARGET
export PATH_TEMP=$PATH_ROOT/output/nbp12-bert/temp/$DATA_FILTER/$NAME_MODEL/bpe/$DATA_SEQUENCE/$DATA_TARGET

python /d/hpc/projects/FRI/up4472/upolanc-thesis/notebook/nbp12-bert.py \
--model_type "$NAME_MODEL" \
--tokenizer_name "$NAME_TOKEN" \
--model_name_or_path "$PATH_BERT" \
--data_dir "$PATH_DATA" \
--output_dir "$PATH_OUTS" \
--cache_dir "$PATH_TEMP" \
--task_name regression_sequence \
--overwrite_output \
--do_train \
--do_eval \
--evaluate_during_training \
--max_seq_length 512 \
--per_gpu_eval_batch_size 32 \
--per_gpu_train_batch_size 32 \
--learning_rate 5e-5 \
--num_train_epochs 250 \
--logging_steps 100 \
--save_steps 25000 \
--warmup_percent 0.1 \
--hidden_dropout_prob 0.1 \
--weight_decay 0.01 \
--n_process 6 \
--optimizer adamw \
--freeze_layers 12 \
--num_features 72 \
--pooler default
