#!/bin/bash

model=${1}
layer=${2}
lang=${3}
split=${4}

og_project_dir=/projects/abeb4417/alignment/awesome-align/
project_dir=/projects/abeb4417/alignment/xlmr-awesome/
model_dir=/rc_scratch/abeb4417/alignment/final_models/alignment_pretraining/
output_dir=${project_dir}layer_analysis_output/${model}_${layer}/
log_dir=${project_dir}layer_analysis_output/logs/

mkdir -p ${output_dir}
mkdir -p ${log_dir}

model_path=${model_dir}/${model}/${lang}/final_model/

awesome-align \
  --output_file=${output_dir}/${lang}_${split}.output \
  --model_name_or_path=${model_path} \
  --data_file=${og_project_dir}alignment_evaluation_data/${split}/${lang}.src-tgt \
  --extraction 'softmax' \
  --tokenizer_name=xlm-roberta-base \
  --batch_size 32 \
  --align_layer ${layer}

python tools/aer.py ${og_project_dir}alignment_evaluation_data/${split}/${lang}.gold ${output_dir}/${lang}_${split}.output --allSure \
  --log_directory ${log_dir}
