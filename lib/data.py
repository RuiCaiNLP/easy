# -*- coding: UTF-8 -*-
from __future__ import division
from collections import Counter
import numpy as np
import os


class Vocab(object):
    PAD, UNK, NUM, FlOAT = 0, 1, 2, 3

    def __init__(self, input_file, pret_file, min_occur_count=1):
        word_counter = Counter()
        tag_set = set()
        rel_set = set()

        content_idx = 0
        with open(input_file) as f:
            for line in f.readlines():
                info = line.strip().split()

                if info:
                    if content_idx == 0:
                        for word in info:
                            word = self.normalize(word)
                            word_counter[word] += 1
                    elif content_idx == 1 or content_idx == 2:
                        for tag in info:
                            tag_set.add(tag)
                    elif content_idx > 7 :
                        for rel in info:
                            rel_set.add(rel)
                    content_idx += 1
                else:
                    content_idx = 0

        self._id2word = ['<PAD>',  '<UNK>', '<NUM>', '<FLOAT>']
        self._id2tag = ['<PAD>', '<UNK>']
        self._id2rel = ['<PAD>']

        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)

        self._pret_file = pret_file
        if pret_file or os.path.exists(pret_file):
            self._add_pret_words(pret_file)

        self._id2tag += list(tag_set)
        self._id2rel += list(rel_set)


        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        self._tag2id = reverse(self._id2tag)
        self._rel2id = reverse(self._id2rel)
        print("Vocab info: #words %d #tags %d #rels %d" % \
              (self.vocab_size,  self.tag_size, self.rel_size))

    def normalize(self, token):
        penn_tokens = {
            '-LRB-': '(',
            '-RRB-': ')',
            '-LSB-': '[',
            '-RSB-': ']',
            '-LCB-': '{',
            '-RCB-': '}'
        }

        if token in penn_tokens:
            return penn_tokens[token]

        token = token.lower()
        try:
            int(token)
            return "<NUM>"
        except:
            pass
        try:
            float(token.replace(',', ''))
            return "<FLOAT>"
        except:
            pass
        return token

    def _add_pret_words(self, pret_file):
        self._words_in_train_data = len(self._id2word)
        print('#words in training set:', self._words_in_train_data)
        words_in_train_data = set(self._id2word)
        with open(pret_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line:
                    word = line[0]
                    if word not in words_in_train_data:
                        continue
                        self._id2word.append(word)
        print 'Total words:', len(self._id2word)

    def get_pret_embs(self, word_dims=100):
        assert (self._pret_file is not None), "No pretrained file provided."
        if self._pret_file is None or not os.path.exists(self._pret_file):
            print "Pretrained embedding randomly initialized."
            return np.random.randn(len(self._id2word), word_dims).astype(np.float32)
        embs = [[]] * len(self._id2word)
        print(len(self._id2word))
        words_in_train_data = set(self._id2word)
        emb_size = 0
        pretrained_used_nums = 0
        with open(self._pret_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line:
                    word, data = line[0], line[1:]
                    if word in words_in_train_data:
                        pretrained_used_nums += 1
                        embs[self._word2id[word]] = data
                        emb_size = len(data)
        print("pretranied words in train set numbers:", pretrained_used_nums)
        for idx, emb in enumerate(embs):
            if not emb:
                embs[idx] = np.zeros(emb_size)
        pret_embs = np.array(embs, dtype=np.float32)
        return pret_embs / np.std(pret_embs)

    def get_flag_embs(self, flags_dims):
        embs = np.random.randn(2, flags_dims).astype(np.float32)
        return embs / np.std(embs)


    def get_word_embs(self, word_dims):
        embs = np.random.randn(self.words_in_train, word_dims).astype(np.float32)
        return embs / np.std(embs)

    def get_tag_embs(self, tag_dims):
        embs = np.random.randn(self.tag_size, tag_dims).astype(np.float32)
        return embs / np.std(embs)

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]



    def rel2id(self, xs):
        if isinstance(xs, list):
            return [self._rel2id[x] for x in xs]
        return self._rel2id[xs]



    def id2rel(self, xs):
        if isinstance(xs, list):
            return [self._id2rel[x] for x in xs]
        return self._id2rel[xs]

    def tag2id(self, xs):
        if isinstance(xs, list):
            return [self._tag2id.get(x, self.UNK) for x in xs]
        return self._tag2id.get(xs, self.UNK)

    @property
    def words_in_train(self):
        return self._words_in_train_data

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def tag_size(self):
        return len(self._id2tag)

    @property
    def rel_size(self):
        return len(self._id2rel)




class DataLoader(object):
    def __init__(self, input_file, vocab):
        sents = []
        sent = []
        self.vocab = vocab
        with open(input_file) as f:
            content_idx = 0
            for line in f.readlines():
                info = line.strip().split()
                if info:
                    if content_idx == 0:
                        words = [vocab.word2id(word.lower()) for word in info]
                        sent.append(words)
                    elif content_idx == 1 or content_idx == 2:
                        tags = [vocab.tag2id(tag) for tag in info]
                        sent.append(tags)
                    elif content_idx == 7:
                        preds = [int(pred) for pred in info]
                        sent.append(preds)
                    elif content_idx > 7 :
                        rels = [vocab.rel2id(rel) for rel in info]
                        sent.append(rels)
                    content_idx += 1
                else:
                    if len(sent) < 5:
                        print("error!")

                        print(vocab.id2word(sent[0]))
                    sents.append(sent)
                    sent = []
                    content_idx = 0

        print("labeled", len(sents))

        samples_num = len(sents)
        self.samples = sents

    @property
    def idx_sequence(self):
        return [x[1] for x in sorted(zip(self._record, range(len(self._record))))]

    def get_batches(self, batch_size, shuffle = True):
        batches = []
        batches_num = int(len(self.samples)/batch_size)
        for idx in range(batches_num):
            batch_samples = self.samples[idx*batch_size: (idx+1)*batch_size]
            batches.append(batch_samples)
        print("log")
        print(len(batches))
        if shuffle:
            np.random.shuffle(batches)

        for batch_samples in batches:
            max_len = 0
            for sample in batch_samples:
                if len(sample[0]) > max_len:
                    max_len = len(sample[0])
            ##what need to be batched: word, pos, rels
            words_inputs = np.zeros((batch_size, max_len), dtype=np.int32)
            tag_inputs = np.zeros((batch_size, max_len), dtype=np.int32)
            pred_indices = []
            rel_targets = []
            for i, sample in enumerate(batch_samples):
                words_inputs[i][:len(sample[0])] = sample[0]
                tag_inputs[i][:len(sample[1])] = sample[1]
                pred_indices.append(sample[3])
                preds_num = len(sample[3])

                for j in range(preds_num):

                    rel_target = np.zeros((max_len), dtype=np.int32)
                    rel_target[:len(sample[4 + j])] = sample[4 + j]
                    rel_targets.append(rel_target)
            #print(words_inputs[0])
            #print(self.vocab.id2word(list(words_inputs[0])))
            yield words_inputs, tag_inputs, pred_indices, rel_targets
