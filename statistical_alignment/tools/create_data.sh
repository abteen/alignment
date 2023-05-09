#! /bin/bash

project_directory=/projects/abeb4417/alignment_pub/

for lang in bzd shp gn quy
do

  #Split ||| separated bitext into separate files
  python tools/tlm_to_bitext.py ${lang} ${project_directory}

  #Join all source and target files
  cd input_data/${lang}
  for suffix in ${lang} es; do
    cat train.${suffix} dev.${suffix} test.${suffix} > all.${suffix}
  done
  cat all.${lang} all.es > all.joint

  #Train the sentencepiece model using the joint data
  spm_train --input_sentence_size      100000000 \
              --model_prefix             bpe.${lang} \
              --model_type               bpe \
              --num_threads              4 \
              --split_by_unicode_script  1 \
              --split_by_whitespace      0 \
              --remove_extra_whitespaces 1 \
              --normalization_rule_name  identity \
              --vocab_size               40000 \
              --character_coverage       1.0 \
              --add_dummy_prefix         1 \
              --input                    all.joint \
              --hard_vocab_limit  False

  #Encode the using the trained model
  for suffix in ${lang} es; do
    for split in train dev "test" traindev traintest; do
      spm_encode --model bpe.${lang}.model < ${split}.${suffix} > ${split}.${suffix}.bpe
    done
  done

  cd ../..

done