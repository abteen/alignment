lang=${1}
model=${2}
chunk=${3}
epochs=${4}

echo ${lang}
echo ${model}
echo ${chunk}
echo ${epochs}

python pretraining/run_singlelang_pretraining.py \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_a100_xlmr.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  experiment_name=alignment_pretraining group_name=mlm_wiki_${model}_chunked-${chunk}_${epochs}. run_name=${lang} check_git_status=True use_wandb=True \
  target_languages=[${lang}] eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
  log_directory="/projects/abeb4417/alignment/logs/" output_directory="/rc_scratch/abeb4417/alignment/final_models/" \
  dev=False data_subset=["mlm_tgt.${lang}","wiki.${lang}"] chunk_dataset=${chunk} training_arguments.num_train_epochs=${epochs} model=${model}