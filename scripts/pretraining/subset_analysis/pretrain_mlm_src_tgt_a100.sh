lang=${1}

echo ${lang}

for chunk in True
do
for n in 50 100 200 400 800 1600 3200 6400 12800 25600
do
actual_n=$((${n} / 2))
echo ${actual_n}
python pretraining/run_multilang_pretraining.py \
  --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
  --training_args configs/training_arguments/pretraining/pretraining_a100.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  experiment_name=subset_analysis_bc group_name=${lang}_mlm-st run_name=mlm-st_${lang}_${n}_${chunk} check_git_status=True use_wandb=True \
  target_languages=[${lang}] eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
  log_directory="/projects/abeb4417/alignment/logs/" output_directory="/rc_scratch/abeb4417/alignment/" \
  dev=False data_sets=[[mlm_tgt.${lang}],[mlm_src.${lang}]] chunk_dataset=${chunk} training_arguments.num_train_epochs=40 model=bert-base-multilingual-cased use_first_n=${actual_n}
done
done