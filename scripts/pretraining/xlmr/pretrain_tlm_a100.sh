lang=${1}

echo ${lang}

python pretraining/run_singlelang_pretraining.py \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_a100.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  experiment_name=alignment_pretraining group_name=xlmr_tlm_80 run_name=${lang} check_git_status=True use_wandb=True \
  target_languages=[${lang}] eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
  log_directory="/projects/abeb4417/alignment/logs/" output_directory="/rc_scratch/abeb4417/alignment/final_models/xlmr/" \
  dev=False data_subset=[tlm.${lang}] chunk_dataset=False training_arguments.num_train_epochs=80 model=xlm-roberta-base