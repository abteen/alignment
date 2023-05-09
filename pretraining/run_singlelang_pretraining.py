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

    if config.get('random_length_analysis', None):
        num_examples_in_group = config['random_length_analysis']

        shuffled_indices = [i for i in range(len(train_dataset))]
        random.seed(config['random_length_analysis_seed'])
        random.shuffle(shuffled_indices)

        first_n = shuffled_indices[:num_examples_in_group]
        train_dataset = train_dataset.select(first_n)

        logging.info('Selecting random subset of length {length} using seed {seed}.'.format(
            length=num_examples_in_group,
            seed=config['random_length_analysis_seed']))

    if config.get('length_analysis', None):

        num_examples_in_group = config['length_analysis']
        length_analysis_group = config['length_analysis_group']

        logging.info('You can make {num_groups_avail} groups with {tot_ex} examples and group size {size}'.format(
            size=num_examples_in_group,
            tot_ex=len(train_dataset),
            num_groups_avail=len(train_dataset) // num_examples_in_group
        ))
        logging.info(
            'Trying to access group {group} with indices {s} --> {e} with total available examples equal to {tot_avail}'.format(
                group=length_analysis_group,
                s=length_analysis_group * num_examples_in_group,
                e=(length_analysis_group + 1) * num_examples_in_group,
                tot_avail=len(train_dataset)
            ))

        if len(train_dataset) // num_examples_in_group <= length_analysis_group:
            logging.info('Group OOB, exiting.')
            sys.exit(1)

        # Calculate number of characters per example
        train_dataset = train_dataset.map(
            lambda example: {'char_length': len(example['text'])}, batched=False
        )

        # Sort dataset based on char length
        train_dataset = train_dataset.sort('char_length')

        selected_indices = [i for i in range(length_analysis_group * num_examples_in_group,
                                             (length_analysis_group + 1) * num_examples_in_group)]

        train_dataset = train_dataset.select(selected_indices)

        tot_chars = 0
        for ex in train_dataset:
            tot_chars += len(ex['text'])

        logging.info('Selected {len} examples, with {chars} total characters. Average chars / example: {avg}'.format(
            len=len(train_dataset),
            chars=tot_chars,
            avg=tot_chars / len(train_dataset)
        ))

        # with open('length_analysis_data/tlm{group}.es'.format(group=length_analysis_group), 'w') as esf, open('length_analysis_data/tlm{group}.quy'.format(group=length_analysis_group), 'w') as quyf:
        #     for ex in train_dataset:
        #         es, tgt = ex['text'].split(' ||| ')
        #         esf.write(
        #             es + '\n'
        #         )
        #
        #         quyf.write(tgt + '\n')
        #
        # raise KeyError

    if config.get('use_first_n', None):

        if config['use_first_n'] > len(train_dataset):
            logging.info('Selected {n} is larger than total train size. Exiting.'.format(n=config['use_first_n']))

        shuffled_indices = [i for i in range(len(train_dataset))]
        random.seed(config['data_seed'])
        random.shuffle(shuffled_indices)

        first_n = shuffled_indices[:config['use_first_n']]
        train_dataset = train_dataset.select(first_n)


        logging.info('Randomly shuffling and selecting first {n} examples for training. Data seed: {seed}'.format(n=config['use_first_n'], seed=config['data_seed']))

    if config['chunk_dataset']:

        # We want to separate each document with the separator token
        train_dataset = train_dataset.map(
            lambda example: {'text': example['text'] + tokenizer.sep_token}, batched=False, num_proc=num_processes
        )

        #Tokenize the examples
        train_dataset = train_dataset.map(
            lambda examples: tokenizer(examples['text'], padding=False, truncation=False, add_special_tokens=False,
                                       return_length=True), batched=True, num_proc=num_processes
        )

        #Sort by length and concatenate all input examples together
        train_dataset = train_dataset.sort('length')

        dset_tensors = [torch.tensor(d) for d in train_dataset['input_ids']]
        all_tensors = torch.cat(dset_tensors, 0)

        #Create our new input examples, all but one are sized to max_length - 2
        examples = torch.split(all_tensors, 254, dim=0)

        train_dataset = Dataset.from_dict({'ids': examples})

        print(train_dataset)
        logging.info('Train dataset before encode_plus: {}'.format(train_dataset[0]))

        #Encode to proper format
        train_dataset = train_dataset.map(
            lambda examples: tokenizer.encode_plus(examples['ids'], add_special_tokens=True, return_length=True),
            batched=False, num_proc=8)

        logging.info('Train dataset example after encode_plus {}'.format(train_dataset[0]))
        logging.info(train_dataset)
        logging.info(set(train_dataset['length']))
        assert all(x <= 256 for x in set(train_dataset['length']))

    else:

        train_dataset = train_dataset.map(
            lambda examples: tokenizer(examples['text'], truncation=True, max_length=256), batched=True,
            remove_columns=['text'],
            num_proc=num_processes)


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
    trainer = Trainer(
        model=model,
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
    logging.info('-'*25 + 'Finished with language: {}'.format(target_languages) + '-'*25)
















