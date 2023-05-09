lang=${1}

echo ${lang}

for group in 0 1 2 3
do
  python pretraining/run_singlelang_pretraining.py \
    --experiment_config configs/experiment/pretraining/pretraining_default.yaml \
    --training_args configs/training_arguments/pretraining/pretraining_a100.yaml \
    --tokenizer_config configs/tokenizer/xlmr_tokenizer.yaml \
    experiment_name=length_analysis group_name=${lang}_tgt_wiki run_name=mlm_tgt_wiki_7508_${group}.${lang} check_git_status=True use_wandb=True \
    target_languages=[${lang}] eval_languages=None data_directory="/projects/abeb4417/data/alignment_data/" \
    log_directory="/projects/abeb4417/alignment/logs/" output_directory="/rc_scratch/abeb4417/alignment/" \
    dev=False chunk_dataset=True training_arguments.num_train_epochs=80 model=bert-base-multilingual-cased \
    data_subset=["mlm_tgt.${lang}","wiki.${lang}"] length_analysis=7508 length_analysis_group=${group}


    # bzd mlm_tgt total characters 207618 total examples 7508
    # gn total groups mlm_tgt and wiki using examples: 4
    # quy total groups mlm_tgt and wiki using examples: 19
done