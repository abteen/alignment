
seed=42

for lang in ay sw
do

python run_finetuning_ner.py \
  --experiment_config configs/experiment/finetuning/finetuning_default.yaml \
  --training_args configs/training_arguments/finetuning/ner_default.yaml \
  --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
  experiment_name=lang_emb_finetune group_name=multilang_mlm_noemb run_name=${lang} check_git_status=True use_wandb=True \
  seed=${seed} \
  source_language=en eval_language=en \
  pretrained_model_checkpoint=/rc_scratch/abeb4417/language_embedding/models/multilang_mlm_noemb/${lang}/final_model/

done