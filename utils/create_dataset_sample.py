import pandas, numpy
from sacremoses import MosesTokenizer

def tok(line):
    tokenizer = MosesTokenizer(lang='es')
    return ' '.join(tokenizer.tokenize(line, escape=False, protected_patterns="'"))

if __name__ == '__main__':

    test_tsv = 'data/anli.mt.test.tsv'

    df = pandas.read_csv(test_tsv, delimiter='\t')

    #Drop any rows which have missing translations, to make sure final sample is parallel
    df = df.drop('id', axis=1).drop('type', axis=1)
    df = df.replace('', numpy.nan)
    nan_values = df[df.isna().any(axis=1)]
    print(nan_values)
    df = df.dropna(axis=0)

    #Filter rows based on rough estimate of length
    df = df[df['Español'].map(lambda x: len(x) > 50 and len(x) < 150)]
    print(df)

    sample = df.sample(n=50, replace=False, random_state=5)
    print(sample)
    sample = sample.applymap(tok)
    print(sample)

    tokenizer = MosesTokenizer(lang='es')

    sample.to_csv('data/alignment_sample/anli_sample.tsv', sep='\t')

    for language in sample.columns:
        sample[['Español', language]].to_csv('data/alignment_sample/languages/{}.tsv'.format(language), sep='\t', index=False)