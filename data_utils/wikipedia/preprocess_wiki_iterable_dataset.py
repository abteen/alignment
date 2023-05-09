from bfml.data_loading import read_lines, chunk_and_tokenize_documents_python, chunk_and_tokenize_documents_for_dataset
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

    data = load_dataset(path=args.input_dir, name=args.run_name, split='train', streaming=True)
    # data = data.filter(lambda ex: len(ex['text']) >= 50)

    data = data.map(
        lambda examples: chunk_and_tokenize_documents_python(examples['text'], tokenizer, 256), batched=True,
        remove_columns=['text'],
        num_proc=num_processes)
        # features=features)

    if args.output_dir:
        data.save_to_disk(os.path.join(args.output_dir, args.run_name))




