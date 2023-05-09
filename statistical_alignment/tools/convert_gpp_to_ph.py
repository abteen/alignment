import sys, os, re

if __name__ == '__main__':

    lang = sys.argv[1]
    split = sys.argv[2]

    input_file = 'final_outputs/{split}/{lang}/{lang}_{split}.VA3.final'.format(split=split, lang=lang)
    output_dir = 'processed_outputs/{split}/{lang}/'.format(split=split, lang=lang)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    with open(input_file, 'r') as gpp:
        gpp_lines = [line.strip() for line in gpp]

    assert len(gpp_lines) % 3 == 0


    with open(output_dir + '{lang}_{split}.output'.format(lang=lang, split=split), 'w') as hyp, open(output_dir + '{lang}_{split}.src_tgt'.format(lang=lang, split=split), 'w') as st:
        for i in range(0, len(gpp_lines), 3):
            subset = gpp_lines[i:i + 3]
            print(subset)

            tgt_sentence = subset[1]
            src_output = subset[2]

            found_alignments = re.findall(r'\({(.*?)}\)', src_output)[1:]
            found_src_tokens = re.findall(r'}\) (.+?) \({', src_output)

            print(src_output)
            print(found_alignments)
            print(found_src_tokens)


            assert len(found_alignments) == len(found_src_tokens)

            extracted_alignment = [] #ith src --> jth tgt

            for i in range(len(found_src_tokens)):
                tgt_align = found_alignments[i].strip()
                if tgt_align:
                    for j in tgt_align.split(' '):
                        extracted_alignment.append('{i}-{j}'.format(i=i,j=int(j)-1))

            print(extracted_alignment)
            print('src: {}'.format(found_src_tokens))
            print('tgt: {}'.format(tgt_sentence.split(' ')))

            hyp.write(' '.join(extracted_alignment) + '\n')
            st.write(' '.join(found_src_tokens) + ' ||| ' + tgt_sentence + '\n')

