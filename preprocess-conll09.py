def read_conll(filename):
    data = []
    sentence = []
    with open(filename, 'r') as fp:
        for line in fp:
            if len(line.strip()) > 0:
                sentence.append(line.strip().split())
            else:
                data.append(sentence)
                sentence = []
        if len(sentence) > 0:
            data.append(sentence)
            sentence = []
    return data




def srl2ptb(origin_data):
    srl_data = []
    for sentence in origin_data:
        one_sample = []
        pred_set = []
        word_set = []
        gold_pos_set = []
        predict_pos_set = []
        gold_dep_label_set = []
        predict_dep_label_set = []
        gold_dep_head_set = []
        predict_dep_head_set = []
        SRL_label_sets = []

        for i in range(len(sentence)):
            word_set.append(sentence[i][1])
            gold_pos_set.append(sentence[i][4])
            predict_pos_set.append(sentence[i][5])
            gold_dep_head_set.append(sentence[i][8])
            predict_dep_head_set.append(sentence[i][9])
            gold_dep_label_set.append(sentence[i][10])
            predict_dep_label_set.append(sentence[i][11])
            if sentence[i][12] == 'Y':
                pred_set.append(str(i))

        one_sample.append(word_set)
        one_sample.append(gold_pos_set)
        one_sample.append(predict_pos_set)
        one_sample.append(gold_dep_head_set)
        one_sample.append(predict_dep_head_set)
        one_sample.append(gold_dep_label_set)
        one_sample.append(predict_dep_label_set)
        one_sample.append(pred_set)
        for arg_idx in range(len(pred_set)):
            SRL_label_set = []
            for i in range(len(sentence)):
                SRL_label_set.append(sentence[i][14+arg_idx])
            one_sample.append(SRL_label_set)
        if len(pred_set) > 0:
            srl_data.append(one_sample)

    return srl_data



def save(srl_data, path):
    with open(path, 'w') as f:
        for sent in srl_data:
            for token in sent:
                f.write('\t'.join(token))
                f.write('\n')
            f.write('\n')


import argparse, os

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', default=None)
    argparser.add_argument('--test', default=None)
    argparser.add_argument('--dev', default=None)
    argparser.add_argument('--out_dir', default='processed')
    argparser.add_argument('--train_fr', default=None)
    argparser.add_argument('--test_fr', default=None)
    argparser.add_argument('--dev_fr', default=None)
    args, extra_args = argparser.parse_known_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    train_conll = read_conll(args.train)
    train_srl = srl2ptb(train_conll)
    save(train_srl, '%s/train_pro' % args.out_dir)

    dev_conll = read_conll(args.dev)
    dev_srl = srl2ptb(dev_conll)
    save(dev_srl, '%s/dev_pro' % args.out_dir)
    os.system('cp %s %s/dev_raw' % (args.dev, args.out_dir))

    test_conll = read_conll(args.test)
    test_srl = srl2ptb(test_conll)
    save(test_srl, '%s/test_pro' % args.out_dir)
    os.system('cp %s %s/test_raw' % (args.test, args.out_dir))


