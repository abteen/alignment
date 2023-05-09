import sys, os

if __name__ == '__main__':

    lang = sys.argv[1]
    project_directory = sys.argv[2]

    file_dir = {
        'train' : os.path.join(project_directory, 'alignment_data/tlm/{lang}/es-{lang}.src_tgt'.format(lang=lang)),
        'dev' : os.path.join(project_directory, 'alignment_evaluation_data/dev/{lang}.src-tgt'.format(lang=lang)),
        'test' : os.path.join(project_directory, 'alignment_evaluation_data/test/{lang}.src-tgt'.format(lang=lang)),
    }

    loaded_files = {}

    for split_name in ['train', 'dev', 'test']:

        input_file = file_dir[split_name]
        tgt_lines = []
        es_lines = []
        with open(input_file, 'r') as tlm:
            for line in tlm:
                line = line.strip()
                if line:
                    if split_name in ['dev', 'test']:
                        tgt, es = line.split(' ||| ')
                    else:
                        es , tgt = line.split(' ||| ')

                    tgt_lines.append(tgt)
                    es_lines.append(es)

            loaded_files[split_name] = {'tgt' : tgt_lines, 'es' : es_lines}

    for split in ['dev', 'test']:
        loaded_files['train{split}'.format(split=split)] = {'tgt' : loaded_files['train']['tgt'] + loaded_files[split]['tgt'],
                                     'es' : loaded_files['train']['es'] + loaded_files[split]['es']}

    output_dir = os.path.join(project_directory, 'statistical_alignment/input_data/{lang}/'.format(lang=lang))

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for split_name,v in loaded_files.items():
        with open(output_dir + '{split}.es'.format(split=split_name), 'w') as es_out, open(output_dir + '{split}.{lang}'.format(split=split_name, lang=lang), 'w') as tgt_out:

            assert len(v['tgt']) == len(v['es'])

            for i in range(len(v['tgt'])):
                es_out.write(v['es'][i] + '\n')
                tgt_out.write(v['tgt'][i] + '\n')

