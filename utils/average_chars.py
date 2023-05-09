import logging, os, torch, random, sys
from pprint import pformat

from bfml.experiment import setup_experiment
from bfml.data_loading import chunk_and_tokenize_documents

from datasets import DatasetDict, concatenate_datasets, load_dataset, Dataset

from transformers import TrainingArguments, BertTokenizer, XLMRobertaConfig, LineByLineTextDataset, AutoTokenizer, XLMRobertaTokenizer
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, EarlyStoppingCallback


def wiki_process(examples):

    cleaned_examples = []
    for example in examples:
        cleaned_examples.append(example.replace('&lt;br&gt;', ' '))

    return {'text' : cleaned_examples}

if __name__ == '__main__':

    args, config, output_directory = setup_experiment()

    model = config.get('model', 'bert-base-multilingual-cased')
    tokenizer_dict = {
        'bert-base-multilingual-cased' : BertTokenizer,
        'xlm-roberta-base' : XLMRobertaTokenizer
    }

    logging.info('Loading tokenizer')
    tok_model = tokenizer_dict[model]
    tokenizer = tok_model.from_pretrained(model)
    logging.info('Tokenizer: {}'.format(pformat(tokenizer)))


    target_languages = config['target_languages']
    eval_languages = config['eval_languages']
    logging.info('Current target languages: {}'.format(target_languages))
    logging.info('Current eval languages: {}'.format(eval_languages))

    train_datasets = []
    eval_datasets = []

    num_processes = int(os.environ['SLURM_NTASKS'])

    for subset in config['data_subset']:
        dataset = load_dataset(config['data_directory'], subset , data_dir=config['data_directory'])
        dataset = dataset['train'] if isinstance(dataset, DatasetDict) else dataset

        if 'wiki' in subset:
            dataset = dataset.map(
                lambda examples: wiki_process(examples['text']), batched=True, num_proc=num_processes
            )

            dataset = dataset.filter(lambda example: len(example['text']) > 10)

        train_datasets.append(dataset)

    train_dataset = concatenate_datasets(train_datasets)


    tot_chars = 0
    for ex in train_dataset:
        tot_chars += len(ex['text'])

    logging.info('Selected {len} examples, with {chars} total characters. Average chars / example: {avg}'.format(
        len=len(train_dataset),
        chars=tot_chars,
        avg=tot_chars / len(train_dataset)
    ))
