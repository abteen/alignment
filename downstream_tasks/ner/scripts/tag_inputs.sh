#!/bin/bash

lang=${1}
input=/projects/abeb4417/alignment/awesome-align/train_alignments_for_pos/inputs/${lang}/es-${lang}.src-tgt
output_dir=/projects/abeb4417/alignment/ner/trained_alignments/tagged_inputs/

python tag_inputs.py ${lang} ${input} ${output_dir}