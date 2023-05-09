#!/bin/bash

lang=${1}
DOWNLOAD_DIR=${2}

date=latest


LANG_DIR=${DOWNLOAD_DIR}/${lang}
mkdir ${LANG_DIR} -p
cd ${LANG_DIR} || exit

wget -nc https://dumps.wikimedia.org/${lang}wiki/${date}/${lang}wiki-${date}-pages-articles-multistream.xml.bz2 --no-check-certificate || exit
bzip2 -dk ${lang}wiki-${date}-pages-articles-multistream.xml.bz2

python -m wikiextractor.WikiExtractor ${lang}wiki-${date}-pages-articles-multistream.xml --processes ${SLURM_NTASKS} --no-templates --json


