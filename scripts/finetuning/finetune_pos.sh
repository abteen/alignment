#! /bin/bash

train_lang=${1}
target_lang=${2}
train_source=${3}
model=${4}

python run_finetuning_pos.py \
--experiment_config configs/experiment/pretraining/pretraining_default.yaml \
--training_args configs/training_arguments/finetuning/pos.yaml \
--tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
experiment_name=pos_finetuning_newbl group_name=bert_base run_name=${train_lang}_${target_lang}_${train_source}_${model} check_git_status=True use_wandb=True dev=False \
 eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
log_directory="/projects/abeb4417/alignment/pos_logs/" output_directory="/rc_scratch/abeb4417/alignment/" \
train_language=${train_lang} target_language=${target_lang} model=${model} alignment_model=${train_source}