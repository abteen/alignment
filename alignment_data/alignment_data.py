import datasets
import os

WIKI_LANGS = ['wiki.' + lang for lang in ['gn', 'quy']]
MLM_LANGS = ['bzd', 'gn', 'quy', 'shp']
MLM_TYPES = ['mlm_src.', 'mlm_tgt.', 'tlm.']

NAMES = WIKI_LANGS + [t + l for t in MLM_TYPES for l in MLM_LANGS]

DESCRIPTIONS = {
    'mlm_tgt': "Monolingual data in the target language, taken from parallel data.",
    'mlm_src': "Monolingual data in the source language (Spanish) taken from parallel data.",
    "tlm": "Aligned source/target data.",
    'wiki': "Monolingual data in the target language taken from Wikipedia."
}

_URLS = {
    'mlm_tgt': 'tlm/',
    'mlm_src': 'tlm/',
    'tlm': 'tlm/',
    'wiki': 'wiki/'
}


class AlignmentDataset(datasets.GeneratorBasedBuilder):


    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name=name,
            description=DESCRIPTIONS[name.split('.')[0]]
        )

        for name in NAMES
    ]

    def _info(self):

        return datasets.DatasetInfo(
            description="A collection of unlabeled data for several Indigenous languages. ",
            features=datasets.Features({
                "text": datasets.Value("string")
            })
        )

    def _split_generators(self, dl_manager):
        source, lang = self.config.name.split('.')
        if source in ['mlm_src', 'mlm_tgt', 'tlm']:
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath': os.path.join(self.config.data_dir, 'tlm/{0}/es-{0}.src_tgt'.format(lang))})]

        elif source in ['wiki']:
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={'filepath' : os.path.join(self.config.data_dir,'wiki/{0}/wiki.{0}.doc.0'.format(lang))})]

    def _generate_examples(self, filepath):
        source, lang = self.config.name.split('.')
        with open(filepath, 'r') as f:
            for key, row in enumerate(f):

                if source == 'mlm_src':
                    yield key, {'text' : row.strip().split(' ||| ')[0]}

                if source == 'mlm_tgt':
                    yield key, {'text' : row.strip().split(' ||| ')[1]}

                if source == 'tlm':
                    yield key, {'text' : row.strip().replace(' ||| ', ' ')}

                if source == 'wiki':
                    yield key, {'text' : row.strip()}


