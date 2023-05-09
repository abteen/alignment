#! /bin/bash

train_lang=${1}
target_lang=${2}
train_source=${3} # 'actual' or name model used to generate alignment
model_name=${4}

model=/rc_scratch/abeb4417/alignment/final_models/alignment_pretraining/${model_name}/${target_lang}/final_model

python run_finetuning_ner.py \
--experiment_config configs/experiment/pretraining/pretraining_default.yaml \
--training_args configs/training_arguments/finetuning/pos.yaml \
--tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
experiment_name=ner_finetuning_newbl group_name=bert_base run_name=${train_lang}_${target_lang}_${train_source}_${model_name} check_git_status=True use_wandb=True dev=False \
 eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
log_directory="/projects/abeb4417/alignment/ner_logs/" output_directory="/rc_scratch/abeb4417/alignment/" \
target_language=${target_lang} train_language=${train_lang} train_source=${train_source} model=${model}