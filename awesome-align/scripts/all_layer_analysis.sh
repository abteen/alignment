#!/bin/bash

for model in tlm_80 mlm_src_tgt_bert-base-multilingual-cased_chunk-True_40 mlm_bert-base-multilingual-cased_chunked-True_80
do
  for lang in bzd gn quy shp
  do
    for layer in {1..12}
    do
    ./scripts/evaluate_layer_analysis.sh ${model} ${layer} ${lang} dev
    done
  done
done


