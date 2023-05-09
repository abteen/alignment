import logging, os, torch, random, sys
from pprint import pformat

from bfml.experiment import setup_experiment
from bfml.data_loading import chunk_and_tokenize_documents
from bfml.metrics import ner_metrics

from datasets import DatasetDict, concatenate_datasets, load_dataset, Dataset
from custom_datasets.conll_pos_dataset import ConllPOSDataset
from custom_datasets.aligned_pos_dataset import ProjectedPOSDataset
from custom_datasets.ner_dataset import NERDataset
from transformers import TrainingArguments, BertTokenizer, XLMRobertaConfig, LineByLineTextDataset, AutoTokenizer, XLMRobertaTokenizer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer, EarlyStoppingCallback


if __name__ == '__main__':

    args, config, output_directory = setup_experiment()

    model = config.get('model', 'bert-base-multilingual-cased')
    tok = config.get('tokenizer', 'bert-base-multilingual-cased')
    tokenizer_dict = {
        'bert-base-multilingual-cased' : BertTokenizer,
        'xlm-roberta-base' : XLMRobertaTokenizer
    }

    logging.info('Loading tokenizer')
    tok_model = tokenizer_dict[tok]
    tokenizer = tok_model.from_pretrained(tok)
    logging.info('Tokenizer: {}'.format(pformat(tokenizer)))


    target_language = config['target_language']
    train_language = config['train_language']
    logging.info('Current train languages: {}'.format(train_language))

    num_processes = int(os.environ['SLURM_NTASKS'])


    train_dataset = NERDataset(lang=train_language, split='train', source=config['train_source'], tokenizer=tokenizer)
    target_dataset = NERDataset(lang=target_language, split='test', source='actual', tokenizer=tokenizer)

    #Load model config and initialize model
    num_unique_labels = len(set(train_dataset.label2id.values()))

    model = AutoModelForTokenClassification.from_pretrained(model, num_labels=num_unique_labels)

    logging.info('Loaded training data from: {}'.format(train_dataset.file))
    logging.info('Loaded evaluation data from: {}'.format(target_dataset.file))
    #Load Training Arguments
    logging.info('Output directory: {}'.format(output_directory))
    training_arguments = TrainingArguments(output_dir=output_directory,
                                           **config['training_arguments'])

    #Load collator for MLM
    collator = DataCollatorForTokenClassification(
        tokenizer=tokenizer,
        padding='longest'
    )


    #Load Trainer
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        data_collator=collator,
        compute_metrics=ner_metrics
    )

    trainer.train()


    model.save_pretrained(os.path.join(output_directory, 'final_model'))
    logging.info('Model saved in: {}'.format(output_directory))
    logging.info('-'*25 + 'Finished with language: {}'.format(target_language) + '-'*25)

    results = trainer.predict(target_dataset)
    results = results.metrics

    logging.info('Final Accuracy:\t{:.2f}'.format(results['test_accuracy']))
    logging.info('Final F1:\t{:.2f}'.format(results['test_f1']))
