from datasets import load_dataset
from transformers import BertTokenizer
import sys

def wiki_process(examples):

    cleaned_examples = []
    for example in examples:
        cleaned_examples.append(example.replace('&lt;br&gt;', ' '))

    return {'text' : cleaned_examples}

if __name__ == '__main__':

    subset = sys.argv[1]
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    dataset = load_dataset('alignment_data/', subset, data_dir='alignment_data/')
    dataset = dataset['train']

    if 'wiki' in subset:
        dataset = dataset.map(
            lambda examples: wiki_process(examples['text']), batched=True
        )

        dataset = dataset.filter(lambda example: len(example['text']) > 10)

    dataset = dataset.map(
        lambda examples: tokenizer(examples['text'], truncation=False, add_special_tokens=False), batched=True,
        remove_columns=['text'])

    print(dataset[0])

    dataset = dataset.map(
        lambda example: {'num_tokens' : len(example['input_ids'])}, batched=False
    )

    print('Subset:\t{}\tTotal Tokens:\t{}\tAverage tokens per example:\t{:.2f}'.format(
        subset,
        sum(dataset['num_tokens']),
        sum(dataset['num_tokens']) / len(dataset)
    ))