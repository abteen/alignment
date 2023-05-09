import torch, json
from conllu import parse_incr


class ProjectedPOSDataset(torch.utils.data.Dataset):
    def __init__(self, lang, split, alignment_model, tokenizer):

        self.tokenizer = tokenizer
        self.max_len = 256
        self.lang = lang
        self.assignment = 'last'
        self.create_label2id()

        file = '/projects/abeb4417/alignment/pos/trained_alignments/projections/{alignment_model}/{lang}.projection'.format(
            lang=lang,
            alignment_model=alignment_model,
            split=split
        )


        self.examples = self.read_file(file)

        self.label_masks = []

        print(self.examples[0])

        print('----------------------------------------')

    def __getitem__(self, idx):
        return self.encode(idx)

    def __len__(self):
        return len(self.examples)

    def read_file(self, file, convert_labels=True):

        inps = []
        self.labels_found = []

        with open(file) as f:
            for line in f:
                inp = json.loads(line.strip())
                toks = inp['target_tokens']
                labels = [self.label2id[tag] for tag in inp['target_tags']]
                self.labels_found.extend(labels)

                assert len(toks) == len(labels)

                inps.append((toks,labels))

        return inps

    def create_label2id(self):

        upos_tags = [
            'ADJ',
            'ADP',
            'PUNCT',
            'ADV',
            'AUX',
            'SYM',
            'INTJ',
            'CCONJ',
            'X',
            'NOUN',
            'DET',
            'PROPN',
            'NUM',
            'VERB',
            'PART',
            'PRON',
            'SCONJ',
            '_'
        ]

        iter = 0
        self.label2id = {}
        for tag in upos_tags:
            self.label2id[tag] = iter
            iter += 1

    def encode(self, id):
        instance = self.examples[id]

        forms = instance[0]
        labels = instance[1]

        label_mask = []

        expanded_labels = []

        for i in range(0, len(forms)):

            subwords = self.tokenizer.tokenize(forms[i])

            if self.assignment == 'first':
                expanded_labels.append(labels[i])
                for j in range(1, len(subwords)):
                    expanded_labels.append(-100)
            elif self.assignment == 'all':
                for j in range(0,len(subwords)):
                    expanded_labels.append(labels[i])
                    if j < len(subwords) - 1:
                        label_mask.append(0)
                    else:
                        label_mask.append(1)

            elif self.assignment == 'last':
                for j in range(0,len(subwords)-1):
                    expanded_labels.append(-100)
                    label_mask.append(0)
                expanded_labels.append(labels[i])
                label_mask.append(1)

        s1 = ' '.join(forms)

        self.label_masks.append(label_mask)


        enc = self.tokenizer(
            s1,
            max_length=self.max_len,
            truncation=True,
            #padding='max_length',
            return_token_type_ids=True,
            #return_tensors='pt',
        )

        if len(expanded_labels) > self.max_len:
            expanded_labels = expanded_labels[:self.max_len]

        enc['labels'] = expanded_labels


        return enc

