lang=${1}
model=${2}
chunk=${3}
epochs=${4}

echo ${lang}
echo ${model}
echo ${chunk}
echo ${epochs}

python pretraining/run_multilang_pretraining.py \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_default.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  experiment_name=alignment_pretraining group_name=mlm_src_tgt_${model}_chunk-${chunk}_${epochs} run_name=${lang} check_git_status=True use_wandb=True \
  target_languages=[${lang}] eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
  log_directory="/projects/abeb4417/alignment/logs/" output_directory="/rc_scratch/abeb4417/alignment/final_models" \
  dev=True data_sets=[[mlm_tgt.${lang}],[mlm_src.${lang}]] chunk_dataset=${chunk} training_arguments.num_train_epochs=${epochs} model=${model}