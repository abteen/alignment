from bfml.data_loading import read_lines, chunk_and_tokenize_documents
from transformers import XLMRobertaTokenizer
import argparse, torch, os


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', default='xlm-roberta-base')
    parser.add_argument('--max_length', default=256)
    parser.add_argument('--run_name')
    parser.add_argument('--input_file')
    parser.add_argument('--output_dir')

    args = parser.parse_args()


    print('Loading tokenizer')
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.tokenizer)


    lines = read_lines(args.input_file)
    input_ids, attention_masks = chunk_and_tokenize_documents(lines, tokenizer, args.max_length)

    torch.save({'input_ids': input_ids, 'attention_masks': attention_masks},
               os.path.join(args.output_dir,
                            args.run_name),
               pickle_protocol=4)

