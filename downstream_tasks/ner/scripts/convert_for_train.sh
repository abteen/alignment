#! /bin/bash

lang=${1}
alignment_method=${2}

input_file=/projects/abeb4417/alignment/ner/trained_alignments/projections/${alignment_method}/${lang}.projection
output_dir=/projects/abeb4417/alignment/ner/trained_alignments/train_data/${alignment_method}/${lang}/

mkdir -p ${output_dir}

python convert_for_train.py ${lang} ${input_file} ${output_dir}train

