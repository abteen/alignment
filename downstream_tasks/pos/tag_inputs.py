import sys, os, stanza, json
from stanza.pipeline.core import DownloadMethod


if __name__ == '__main__':

    lang = sys.argv[1]
    input_file = sys.argv[2]
    output_dir = sys.argv[3]

    with open(input_file, 'r') as inp:
        input = [line.strip() for line in inp]

    nlp = stanza.Pipeline('es', processors='tokenize,pos', tokenize_pretokenized=True, download_method=DownloadMethod.REUSE_RESOURCES)

    spanish_documents = [ex.split(' ||| ')[0].split(' ') for ex in input]
    tgt_documents = [ex.split(' ||| ')[1].split(' ') for ex in input]

    tagged_docs = nlp(spanish_documents)

    with(open(output_dir + '{lang}.tagged'.format(lang=lang), 'w')) as outf:
        for i, sentence in enumerate(tagged_docs.sentences):
            tokens = [word.text for word in sentence.words]
            tags = [word.upos for word in sentence.words]

            outf.write('{}\n'.format(json.dumps({
                'tokens': tokens,
                'tags' : tags,
                'target' : tgt_documents[i]
            })))



