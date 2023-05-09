#! /bin/bash

model=${1}
alignment_model=${2}
model_path=/rc_scratch/abeb4417/alignment/final_models/alignment_pretraining/${model}/gn/final_model/

python run_finetuning_pos.py \
--experiment_config configs/experiment/pretraining/pretraining_default.yaml \
--training_args configs/training_arguments/finetuning/pos.yaml \
--tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
experiment_name=pos_finetuning group_name=adapted_aligned_bert run_name=${alignment_model}_adadpted_aligned_tlm_80mod_es_gn check_git_status=True use_wandb=True dev=False \
 eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
log_directory="/projects/abeb4417/alignment/pos_logs/" output_directory="/rc_scratch/abeb4417/alignment/" \
train_language='gn' target_language='gn' model=${model_path} alignment_model=${alignment_model}