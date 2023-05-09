#!/bin/bash

lang=${1}
alignment_method=${2}

tagged_input=/projects/abeb4417/alignment/awesome-align/train_alignments_for_pos/tagged_inputs/${lang}.tagged
alignment_file=/projects/abeb4417/alignment/fast_align_stuff/alignment-scripts/${alignment_method}-test/trainonly.${lang}.union.talp
output_dir=/projects/abeb4417/alignment/awesome-align/train_alignments_for_pos/projections/${alignment_method}/

log_dir=${output_dir}/logs/

mkdir -p ${output_dir}
mkdir -p ${log_dir}

python project_alignment.py ${lang} ${tagged_input} ${alignment_file} ${output_dir} True > ${log_dir}/${lang}