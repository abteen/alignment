#!/bin/bash

lang=${1}
alignment_method=${2}

tagged_input=/projects/abeb4417/alignment/ner/trained_alignments/tagged_inputs/${lang}.tagged
alignment_file=/projects/abeb4417/alignment/awesome-align/train_alignments_for_pos/${alignment_method}/${lang}.output
output_dir=/projects/abeb4417/alignment/ner/trained_alignments/projections/${alignment_method}/

log_dir=/projects/abeb4417/alignment/ner/trained_alignments/projections/logs/${alignment_method}

mkdir -p ${output_dir}
mkdir -p ${log_dir}

python project_alignment.py ${lang} ${tagged_input} ${alignment_file} ${output_dir} > ${log_dir}/${lang}