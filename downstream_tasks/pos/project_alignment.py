import sys, json
from collections import defaultdict, Counter

if __name__ == '__main__':

    lang = sys.argv[1]
    tagged_inputs = sys.argv[2]
    alignment_file = sys.argv[3]
    output_dir = sys.argv[4]
    reverse = sys.argv[5]

    with open(tagged_inputs, 'r') as f:
        inputs = [json.loads(line.strip()) for line in f]

    with open(alignment_file, 'r') as af:
        alignments = [line.strip() for line in af]

    assert len(inputs) == len(alignments)

    #First create source side tag distribution for each type

    type_tag_dictionary = defaultdict(list)

    for i in range(len(inputs)):

        input = inputs[i]
        alignment = alignments[i]

        if len(alignment) < 1:
            continue

        for ij in alignment.split(' '):
            i,j = ij.split('-')
            if reverse:
                j,i = i,j
            i = int(i)
            j = int(j)

            tgt_type = input['target'][j]
            type_tag_dictionary[tgt_type].append(input['tags'][i])

    type_tag_counter = {k: Counter(v) for k,v in type_tag_dictionary.items()}

    # Now we can do the projection
    x_counter = 0
    x_toks = []

    output_for_file = []

    for i in range(len(inputs)):
        input = inputs[i]
        alignment = alignments[i]
        if len(alignment) < 1:
            continue

        target_tags = [None for _ in range(len(input['target']))]

        for ij in alignment.split(' '):
            i,j = ij.split('-')
            i = int(i)
            j = int(j)

            if reverse:
                j,i = i,j

            print(i,j)

            #Have not projected a tag for this token yet
            if target_tags[j] is None:
                target_tags[j] = input['tags'][i]

            #Have already projected a tag, need to pick the most likely
            else:

                previous_tag = target_tags[j]
                new_tag = input['tags'][i]

                target_type = input['target'][j]

                print('Counter for {type}: {counter}'.format(type=target_type, counter=type_tag_counter[target_type]))

                previous_tag_freq = type_tag_counter[target_type][previous_tag]
                new_tag_freq = type_tag_counter[target_type][new_tag]

                tag_to_keep = new_tag if new_tag_freq > previous_tag_freq else previous_tag

                target_tags[j] = tag_to_keep

            print(target_tags)
        print('------Filling in tags for unaligned tokens------')

        for i, tgt_tag in enumerate(target_tags):

            if tgt_tag is None:

                tgt_tok = input['target'][i]
                try:
                    target_tags[i] = type_tag_counter[tgt_tok].most_common(1)[0][0]
                except KeyError as e:
                    #These are words which have never been aligned to in the whole corpus. Can either assign them 'X'
                    # Or assign them to the most frequent global tag.

                    x_counter += 1
                    x_toks.append(tgt_tok)
                    target_tags[i] = 'X'

        print('Final projected tags: {}'.format(target_tags))
        print('----------------------------------------------')

        output_for_file.append({
            'source_tokens' : input['tokens'],
            'source_tags' : input['tags'],
            'target_tokens' : input['target'],
            'target_tags' : target_tags,
            'alignment' : alignment,

        })



    print('Total number of tokens which have "X" projections: {}'.format(x_counter))
    print('Unique tokens: {}'.format(len(set(x_toks))))


    with open(output_dir + '{lang}.projection'.format(lang=lang), 'w') as outf:
        for out in output_for_file:
            outf.write('{js}\n'.format(js=json.dumps(out)))
