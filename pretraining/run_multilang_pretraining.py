import logging, os, torch, random
from pprint import pformat


from bfml.experiment import setup_experiment
from bfml.samplers import get_language_probabilities_list
from bfml.trainers import TrainerBase, TrainerWithSampler
from bfml.combine import interleave_datasets_cycled
from bfml.data_loading import chunk_dataset_full

from datasets import DatasetDict, Dataset, concatenate_datasets, load_dataset

from torch.utils.data import SequentialSampler




from transformers import TrainingArguments, BertTokenizer, XLMRobertaConfig, LineByLineTextDataset, XLMRobertaTokenizer
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

    num_processes = int(os.environ['SLURM_NTASKS'])

    for data_set in config['data_sets']:
        set_list = []
        for subset in data_set:
            dataset = load_dataset(config['data_directory'], subset, data_dir=config['data_directory'])
            dataset = dataset['train'] if isinstance(dataset, DatasetDict) else dataset

            if 'wiki' in subset:
                dataset = dataset.map(
                    lambda examples: wiki_process(examples['text']), batched=True, num_proc=num_processes
                )

                dataset = dataset.filter(lambda example: len(example['text']) > 10)

            if config.get('use_first_n', None):

                if config['use_first_n'] > len(dataset):
                    logging.info(
                        'Selected {n} is larger than total train size. Exiting.'.format(n=config['use_first_n']))

                shuffled_indices = [i for i in range(len(dataset))]
                random.seed(config['data_seed'])
                random.shuffle(shuffled_indices)

                first_n = shuffled_indices[:config['use_first_n']]
                dataset = dataset.select(first_n)

                logging.info(
                    'Randomly shuffling and selecting first {n} examples for training from source {dataset}. Data seed: {seed}'.format(
                        n=config['use_first_n'], seed=config['data_seed'], dataset=data_set))

            if config['chunk_dataset']:
                dataset = chunk_dataset_full(dataset, tokenizer)

            else:

                dataset = dataset.map(
                    lambda examples: tokenizer(examples['text'], truncation=True, max_length=256), batched=True,
                    remove_columns=['text'],
                    num_proc=num_processes)

            set_list.append(dataset)

        set_dataset = concatenate_datasets(set_list)

        train_datasets.append(set_dataset)


    lengths = [dataset.__len__() for dataset in train_datasets]

    logging.info('Total number of all examples from all sources: {final_n}'.format(final_n=sum(lengths)))


    probabilities = get_language_probabilities_list(lengths)

    train_dataset = interleave_datasets_cycled(datasets=train_datasets,
                                               languages=target_languages,
                                               probabilities=probabilities,
                                               seed=config.data_seed,
                                               batch_size=config.expected_batch_size)

    for col_name in ['ids', 'length']:
        if col_name in train_dataset.features:
            train_dataset = train_dataset.remove_columns(col_name)


    sampler = SequentialSampler(train_dataset)


    #Load model config and initialize model
    model = AutoModelForMaskedLM.from_pretrained(model)

    logging.info('Final loaded training data: {}'.format(train_dataset))


    #Load Training Arguments
    logging.info('Output directory: {}'.format(output_directory))
    training_arguments = TrainingArguments(output_dir=output_directory,
                                           **config['training_arguments'])

    #Load collator for MLM
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)


    callbacks = []
    oa = config.get('other_arguments', None)
    if oa:
        early_stop = oa.get('early_stopping_patience')
        if early_stop:
            callbacks = [EarlyStoppingCallback(early_stopping_patience=config['other_arguments'].early_stopping_patience)]

    #Load Trainer
    trainer = TrainerWithSampler(
        model=model,
        sampler=sampler,
        args=training_arguments,
        train_dataset=train_dataset,
        data_collator=collator,
        callbacks=callbacks
    )

    if os.path.isdir(output_directory) and any('checkpoint-' in files for files in os.listdir(output_directory)):
        logging.info('Resuming training from checkpoint...')
        trainer.train(resume_from_checkpoint=config.check_resume_training)
    else:
        logging.info('Training from beginning...')
        trainer.train()

    model.save_pretrained(os.path.join(output_directory, 'final_model'))
    logging.info('Model saved in: {}'.format(output_directory))
    logging.info('-' * 25 + 'Finished with language: {}'.format(target_languages) + '-' * 25)
















