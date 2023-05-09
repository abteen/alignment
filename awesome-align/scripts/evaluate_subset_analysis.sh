#!/bin/bash

model=${1}
lang=${2}
split=${3}

project_dir=/projects/abeb4417/alignment/
model_dir=/rc_scratch/abeb4417/alignment/subset_analysis_bc/
output_dir=${project_dir}awesome-align/subset_analysis_bc_output/${model}/
log_dir=${project_dir}awesome-align/subset_analysis_bc_output/logs/

mkdir -p ${output_dir}
mkdir -p ${log_dir}

model_path=${model_dir}/${lang}_wiki/${model}/final_model/

awesome-align \
  --output_file=${output_dir}/${lang}_${split}.output \
  --model_name_or_path=${model_path} \
  --data_file=${project_dir}awesome-align/alignment_evaluation_data/${split}/${lang}.src-tgt \
  --extraction 'softmax' \
  --tokenizer_name=bert-base-multilingual-cased \
  --batch_size 32

python tools/aer.py ${project_dir}alignment_evaluation_data/${split}/${lang}.gold ${output_dir}/${lang}_${split}.output --allSure \
  --log_directory ${log_dir}
