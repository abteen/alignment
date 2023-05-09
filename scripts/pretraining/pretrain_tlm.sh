lang=${1}
model=${2}

echo ${lang}

python pretraining/run_singlelang_pretraining.py \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_default.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  experiment_name=alignment_pretraining group_name=tlm_80 run_name=${lang} check_git_status=True use_wandb=True \
  target_languages=[${lang}] eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
  log_directory="/projects/abeb4417/alignment/logs/" output_directory="/rc_scratch/abeb4417/alignment/final_models/" \
  dev=False data_subset=[tlm.${lang}] chunk_dataset=False training_arguments.num_train_epochs=80 model=${model}