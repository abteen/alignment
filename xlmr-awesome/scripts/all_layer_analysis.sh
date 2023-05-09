#!/bin/bash

for model in mlm_xlm-roberta-base_chunked-True_80 mlm_src_tgt_xlm-roberta-base_chunk-True_40 xlmr_tlm_80
do
  for lang in bzd shp gn quy
  do
    for layer in {1..12}
    do
    ./scripts/evaluate_layer_analysis.sh ${model} ${layer} ${lang} test
    done
  done
done


