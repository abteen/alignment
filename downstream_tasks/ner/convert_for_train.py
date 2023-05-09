import sys, json

if __name__ == '__main__':

    lang = sys.argv[1]
    input_file = sys.argv[2]
    output_file = sys.argv[3]

    with open(input_file, 'r') as inf, open(output_file, 'w') as outf:

        for line in inf:
            input = json.loads(line.strip())

            target_tokens = input['target_tokens']
            target_tags = input['target_tags']

            assert (len(target_tokens) == len(target_tags))

            for i,tok in enumerate(target_tokens):
                outf.write('{lang}:{tok}\t{tag}\n'.format(
                    lang=lang,
                    tok=tok,
                    tag=target_tags[i]
                ))

            outf.write('\n')
