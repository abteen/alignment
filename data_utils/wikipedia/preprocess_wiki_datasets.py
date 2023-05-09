from bfml.data_loading import read_lines, chunk_and_tokenize_documents_python
from transformers import XLMRobertaTokenizer
from datasets import load_dataset, Features, Sequence, Value, concatenate_datasets
import argparse, torch, os

from timeit import default_timer as timer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default='xlm-roberta-base')
    parser.add_argument('--max_length', default=256)
    parser.add_argument('--run_name')
    parser.add_argument('--input_dir')
    parser.add_argument('--output_dir')

    args = parser.parse_args()


    print('Loading tokenizer')
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer)

    num_processes = int(os.environ['SLURM_NTASKS'])

    print('{} processes Available'.format(num_processes))

    if os.path.isdir(args.input_dir):
        data_files = sorted([os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir)])
    else:
        data_files = [args.input_dir]

    created_datasets = []
    for i, data_file in enumerate(data_files):
        data = load_dataset('text', data_files=data_file, name=args.run_name)
        data = data.filter(lambda ex: len(ex['text']) >= 50)

        data = data.map(
            lambda examples: chunk_and_tokenize_documents_python(examples['text'], tokenizer, 256), batched=True,
            remove_columns=['text'],
            num_proc=num_processes)


        created_datasets.append(data)
        print('.................Finished with dataset {} of {}'.format(i, len(data_files)))

    created_datasets = [x['train'] for x in created_datasets]
    print('concating:')
    final_dataset = concatenate_datasets(created_datasets)


    if args.output_dir:
        final_dataset.save_to_disk(os.path.join(args.output_dir, args.run_name))




