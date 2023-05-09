#!/bin/bash

for model in mlm_xlm-roberta-base_chunked-True_80 mlm_src_tgt_xlm-roberta-base_chunk-True_40 xlmr_tlm_80
do
  for lang in shp
  do
    ./scripts/evaluate.sh ${model} ${lang} test
  done
done


