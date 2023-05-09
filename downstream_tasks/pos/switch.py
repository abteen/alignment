

if __name__ == '__main__':

    for lang in ['bzd', 'gn', 'shp', 'quy']:

        input = '/mnt/c/Users/persi/PycharmProjects/alignment/alignment_data/tlm/{lang}/es-{lang}.src_tgt'.format(lang=lang)
        with open(input, 'r') as f:
            lines = [line.strip() for line in input]

        output = '/mnt/c/Users/persi/PycharmProjects/alignment/pos/inputs/{lang}-es.src_tgt'.format(lang=lang)
        with open(output, 'w') as f:
            for line in lines:
                es, tgt = line.split(' ||| ')
                f.write('{tgt} ||| {es}\n'.format(tgt=tgt, es=es))

