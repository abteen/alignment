# Meeting the Needs of Low-Resource Languages: The Value of Automatic Alignments via Pretrained Models

## **Evaluation Data**

Currently, please send us an email for access to the gold annotations. 

## **Running the code**

### **Creating Alignments with Awesome-Align**


**(Optional) Adapting Multilingual Models**

The following arguments are necessary for starting the pretraining scripts:

1. `[LANG]` can be one of `[bzd, gn, quy, shp]`
2. `[MODEL]` in this step should be either `bert-base-multilingual-cased` or `xlm-roberta-base to reproduce paper results, or the name of the multilingual model you would like to adapt
3. `[CHUNK]` can be one of `[True, False]` depending on if you want to chunk the dataset examples together to create packed sequences of size `max_seq_len`, or if each line in the dataset should correspond to a single training example. Recommended value is `True`. 
4. `[EPOCHS]` is the total number of training epochs. 

The parameters used for the variants presented in the paper can be found in the Appendix.

Use the following scripts for adaptation:

1. MLM-T : `scripts/pretraining/pretrain_mlm_tgt.sh [LANG] [MODEL] [CHUNK] [EPOCHS]`
2. MLM-ST : `scripts/pretraining/pretrain_mlm_src_tgt.sh [LANG] [MODEL] [CHUNK] [EPOCHS]`
3. MLM-TLM : `scripts/pretraining/pretrain_tlm.sh [LANG] [MODEL]`
4. MLM-TW : `scripts/pretraining/pretrain_tgt_wiki.sh [LANG] [MODEL] [CHUNK] [EPOCHS]`

To adapt XLM-R, use the scripts in `scripts/xlmr/`.

**Extracting Alignments**

To extract alignments from mBERT, use the scripts found in `awesome_align/scripts/`.

`evaluate.sh` will extract alignments and evaluate them, saving results to the `log_directory`. `[MODEL]` is the directory of the adapted model, or name of the baseline model. `[SPLIT]` is one of `[dev, test]` and `[LANG]` is one of `[bzd, gn, quy, shp]`. The generated alignments which were used for the main results of the paper can be found in `awesome_align/final_outputs/` with scores in `awesome_align/final_outputs/logs/`.


### **Creating Alignments with Statistical Methods**

The statistical alignments can be generated from the `statistical_alignment` folder. We base our implementation off of [this repository](https://github.com/lilt/alignment-scripts). Run `tools/create_data.sh` to prepare the training data. To run fast_align, use `scripts/run_fast_align.sh` and to run Giza++, use `scripts/run_giza.sh`. For each script, (1) modify the folder name on lines 15 and 16 to set the output directory for the result and generated alignments (2) set the project directory. Results may be slightly different from the paper results due to the initialization. 

### **Using Alignments for Projection**

The data and code used for the extrinsic evaluations can be found in `downstream_tasks`. The general steps to reproduce the experiment is to first tag the Spanish side sentences with Stanza using `tag_inputs.py`. The tags and then projected using `project_alignment.py`. Sample scripts to show how to use these files can be found in the corresponding `scripts` folder for each downstream task. For NER, the projected annotations need to be convered into a different format using `convert_for_train.py`, while the POS projections can be directly read by the PyTorch Dataset found in `datasets/aligned_pos_dataset.py`. 

Once the training data is created, you can use the scripts found in `scripts/finetuning/` to finetune mBERT on this data.  

### **Analysis Experiments**
The training data used for the analysis experiments can be found in `data_for_analysis/`. The `orig` subset is just training data, while the `testconcat` version includes the evalution data for FastAlign. To gather FastAlign results, use `statistical_alignment/alignment-scripts/run_fastalign_{length,subset}.sh`. For mBERT, see `scripts/pretraining/{length,subset}_analysis/`

## **Citation**

```
@inproceedings{ebrahimi-etal-2023-meeting,
    title = "{M}eeting the {N}eeds of {L}ow-{R}esource {L}anguages: {T}he {V}alue of {A}utomatic {A}lignments via {P}retrained {M}odels",
    author = "Ebrahimi, Abteen  and
      McCarthy, Arya D.  and
      Oncevay, Arturo  and
      Ortega, John  and
      Chiruzzo, Luis  and
      Gim{\'e}nez-lugo, Gustavo  and
      Coto-solano, Rolando  and
      Kann, Katharina",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.280",
    pages = "3894--3908",
    abstract = "Large multilingual models have inspired a new class of word alignment methods, which work well for the model{'}s pretraining languages. However, the languages most in need of automatic alignment are low-resource and, thus, not typically included in the pretraining data. In this work, we ask: How do modern aligners perform on unseen languages, and are they better than traditional methods? We contribute gold-standard alignments for Bribri{--}Spanish, Guarani{--}Spanish, Quechua{--}Spanish, and Shipibo-Konibo{--}Spanish. With these, we evaluate state-of-the-art aligners with and without model adaptation to the target language. Finally, we also evaluate the resulting alignments extrinsically through two downstream tasks: named entity recognition and part-of-speech tagging. We find that although transformer-based methods generally outperform traditional models, the two classes of approach remain competitive with each other.",
}
```



