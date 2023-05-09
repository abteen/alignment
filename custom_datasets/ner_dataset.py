import torch



class NERDataset(torch.utils.data.Dataset):
    def __init__(self, lang, split, source, tokenizer):

        self.tokenizer = tokenizer

        self.max_len = 256
        self.assignment = 'last'
        self.lang = lang

        self.create_label2id()

        if source == 'actual':
            file = '/projects/abeb4417/alignment/data/ner_data/{lang}/{split}'.format(lang=lang, split=split)
        else:
            file = '/projects/abeb4417/alignment/ner/trained_alignments/train_data/{source}/{lang}/{split}'.format(
                source=source,
                lang=lang,
                split=split
            )

        self.file = file

        self.examples = self.read_file(file)

        print(self.examples[0])
        print('----------------------------------------')

    def __getitem__(self, idx):
        return self.encode(idx)

    def __len__(self):
        return len(self.examples)

    def create_label2id(self):

        ner_tags = [
            'B-ORG',
            'I-ORG',
            'B-PER',
            'I-PER',
            'B-MISC',
            'I-MISC',
            'B-LOC',
            'I-LOC',
            'O'
        ]

        iter = 0
        self.label2id = {}
        for tag in ner_tags:
            self.label2id[tag] = iter
            iter += 1

        self.label2id['E-ORG'] = self.label2id['I-ORG']
        self.label2id['E-PER'] = self.label2id['I-PER']
        self.label2id['E-MISC'] = self.label2id['I-MISC']
        self.label2id['E-LOC'] = self.label2id['I-LOC']

        for ent in ['ORG', 'PER', 'MISC', 'LOC']:
            for conv in [('E', 'I'), ('S', 'B')]:
                self.label2id[conv[0] + '-' + ent] = self.label2id[conv[1] + '-' + ent]


    def read_file(self, file, convert_labels=True):

        inps = []

        with open(file, 'r') as f:
            temp_tokens = []
            temp_labels = []
            for line in f:
                if line.strip():

                    token = line.strip().split('\t')
                    assert len(token) == 2

                    if convert_labels:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(self.label2id[token[1]])

                    else:
                        temp_tokens.append(token[0].replace(self.lang + ':', ''))
                        temp_labels.append(token[1])

                else:
                    inps.append((temp_tokens,temp_labels))
                    temp_tokens = []
                    temp_labels = []
        return inps

    def encode(self, id):
        instance = self.examples[id]


        forms = instance[0]
        labels = instance[1]

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
                expanded_labels.append(labels[i])


        s1 = ' '.join(forms)

        enc = self.tokenizer(
            s1,
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True,
        )



        if len(expanded_labels) > self.max_len:
            expanded_labels = expanded_labels[:self.max_len]

        enc['labels'] = expanded_labels

        return enc


if __name__ == '__main__':

    # x = NERDataset(
    #     file='/projects/abeb4417/lrl/data/ner/rahimi_output/eng/train',
    #     max_len=256,
    #     tokenizer=None,
    #     assignment='last'
    # )

    inps = []
    labels_found = []
    lang='en'
    with open('/projects/abeb4417/lrl/data/ner/rahimi_output/en/train') as f:
        temp_tokens = []
        for line in f:
            if line.strip():
                token = line.strip().split('\t')
                assert len(token) == 2
                temp_tokens.append(
                    (token[0].replace(lang + ':', ''), token[1])
                )
            else:
                inps.append(temp_tokens)
                temp_tokens = []

    print(inps[5])
    print(len(inps))



