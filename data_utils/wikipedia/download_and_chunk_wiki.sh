#! /bin/bash

lang=${1}
download_dir=/rc_scratch/abeb4417/wiki_downloads/
document_dir=/projects/abeb4417/data/wiki_documents
output_dir=/projects/abeb4417/data/chunked_wiki/xlmr/

mkdir -p ${download_dir}
mkdir -p ${document_dir}
mkdir -p ${output_dir}



#Download latest wikipedia and save it to the download directory
./download_wiki.sh ${lang} ${download_dir}

echo "Done Downloading, creating samples from JSON..."

#Convert the json files saved by WikiExtractor to text files, one document per line
python create_wiki_samples.py --input_directory ${download_dir}/${lang} \
                              --output_directory ${document_dir} \
                              --language ${lang} \
                              --output_type document

echo "Done writing samples to file. Starting to chunk"

python preprocess_wiki_datasets.py --tokenizer xlm-roberta-base \
                          --run_name wiki.${lang}.hf \
                          --output_dir ${output_dir} \
                          --input_dir ${document_dir}/${lang}


echo "Done with chunking"


