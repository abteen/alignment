#! /bin/bash

for lang in gn
do
#   # All zero shot models
#   ./scripts/finetuning/finetune_ner.sh en ${lang} actual bert-base-multilingual-cased
#   ./scripts/finetuning/finetune_ner.sh es ${lang} actual bert-base-multilingual-cased
#
#   # Train base model with projected ner data
#   for projection_method in bert-baseline fast_align mlm_bert-base-multilingual-cased_chunked-True_80 mlm_src_tgt_bert-base-multilingual-cased_chunk-True_40 tlm_80
#   do
#        ./scripts/finetuning/finetune_ner.sh ${lang} ${lang} ${projection_method} bert-base-multilingual-cased
#    done

    #Supervised models
#   ./scripts/finetuning/finetune_ner.sh ${lang} ${lang} actual bert-base-multilingual-cased
#   ./scripts/finetuning/finetune_ner.sh ${lang} ${lang} actual bert-base-multilingual-cased

  # Train adapted model model with real ner data
#   for projection_method in mlm_bert-base-multilingual-cased_chunked-True_80 mlm_src_tgt_bert-base-multilingual-cased_chunk-True_40 tlm_80
#   do
#        ./scripts/finetuning/finetune_ner_with_model.sh en ${lang} actual ${projection_method}
#        ./scripts/finetuning/finetune_ner_with_model.sh es ${lang} actual ${projection_method}
#    done

  # Train adapted model with projected ner data
  for projection_method in mlm_bert-base-multilingual-cased_chunked-True_80 mlm_src_tgt_bert-base-multilingual-cased_chunk-True_40 tlm_80
   do
        ./scripts/finetuning/finetune_ner_with_model.sh gn ${lang} ${projection_method} mlm_bert-base-multilingual-cased_chunked-True_80
    done
done