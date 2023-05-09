from datasets import load_dataset, Dataset
from datasets import set_caching_enabled
from transformers import BertTokenizer, XLMRobertaTokenizer

import torch


if __name__ == '__main__':

    set_caching_enabled(False)
    dataset = load_dataset('/projects/abeb4417/data/alignment_data/', 'wiki.quy', data_dir='/projects/abeb4417/data/alignment_data/', split='train')
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    dataset = dataset.filter(lambda example: len(example['text']) > 10)

    dataset = dataset.map(
        lambda example: {'text' : example['text'] + tokenizer.sep_token}, batched=False, num_proc=8
    )

    dataset = dataset.map(
        lambda examples: tokenizer(examples['text'], padding=False, truncation=False, add_special_tokens=False, return_length=True), batched=True, num_proc=8
    )

    dataset = dataset.sort('length')

    tens = [torch.tensor(d) for d in dataset['input_ids']]

    all = torch.cat(tens, 0)

    examples = torch.split(all, 254, dim=0)

    dataset = Dataset.from_dict({'ids' : examples})

    print(dataset[0])
    dataset = dataset.map(lambda examples: tokenizer.encode_plus(examples['ids'], add_special_tokens=True, return_length=True), batched=False, num_proc=8)

    print(dataset[0])
    print(set(dataset['length']))
