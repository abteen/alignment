import sys, os, json, argparse


def process_document(text):
    return [text.replace('\n', ' ')]

def process_text(text):
    return text.split('\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_directory')
    parser.add_argument('--chunk_size', default=100000)
    parser.add_argument('--language')
    parser.add_argument('--output_directory')
    parser.add_argument('--output_type', help="Set as 'document' to print a single document per line, "
                                              "or set as 'section' to print one section per line "
                                              "(do not preserve document structure)")


    args = parser.parse_args()

    language_dir = os.path.join(args.input_directory, 'text')
    wiki_folders = os.listdir(language_dir)
    wiki_folders.sort()

    doc_counter = 0
    processed_lines = []

    process_function = process_document if args.output_type == 'document' else process_text

    for folder in wiki_folders:
        folder_path = os.path.join(language_dir, folder)
        wiki_files = [os.path.join(folder_path, file_name) for file_name in os.listdir(folder_path)]
        wiki_files.sort()

        for wiki_file in wiki_files:
            with open(os.path.join(language_dir, wiki_file), 'r') as f:
                for line in f.readlines():
                    doc_counter += 1
                    # Process wiki raw text dumps, preserving documents if necessary
                    line = line.strip()
                    line_data = json.loads(line)

                    if line_data['text']:
                        processed_lines.extend(process_function(line_data['text']))


    print('Loaded {} documents out of {} total read documents'.format(len(processed_lines), doc_counter))

    if args.output_directory:

        output_dir = os.path.join(args.output_directory, args.language)

        os.makedirs(output_dir, exist_ok=True)

        output_type = 'doc' if args.output_type == 'document' else 'sec'

        counter = 0

        chunk_size = min(args.chunk_size, len(processed_lines))
        print(chunk_size)
        for i in range(0,len(processed_lines) - chunk_size + 1, chunk_size):

            with open(os.path.join(output_dir, 'wiki.{}.{}.{}'.format(args.language, output_type, counter)), 'w') as f:
                for line in processed_lines[i:i+chunk_size]:
                    f.write(line + '\n')

            counter += 1