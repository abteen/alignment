import random

if __name__ == '__main__':

    with open('alignment_data/tlm/quy/es-quy.src_tgt', 'r') as f:
        tlm = [line.strip() for line in f]

    random.seed(42)
    indices = [i for i in range(len(tlm))]

    random.shuffle(indices)

    for n in [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 121064]:
        with open('subset_analysis_data/train{n}.es'.format(n=n), 'w') as esf, open('subset_analysis_data/train{n}.quy'.format(n=n), 'w') as quyf:
            for index in indices[:n]:
                es, quy = tlm[index].split(' ||| ')

                esf.write('{}\n'.format(es))
                quyf.write('{}\n'.format(quy))