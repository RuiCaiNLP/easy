# -*- coding: UTF-8 -*-
from __future__ import division
from collections import Counter
import numpy as np
import os


class Vocab(object):
    PAD, DUMMY, UNK = 0, 1, 2

    def __init__(self, input_file, pret_file, min_occur_count=2):
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

        self._id2word = ['<PAD>', '<DUMMY>', '<UNK>']
        self._id2tag = ['<PAD>', '<DUMMY>', '<UNK>']
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
        with open(self._pret_file) as f:
            for line in f.readlines():
                line = line.strip().split()
                if line:
                    word, data = line[0], line[1:]
                    if word in words_in_train_data:
                        embs[self._word2id[word]] = data
                        emb_size = len(data)

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
    def __init__(self, input_file, n_bkts, vocab):
        sents = []
        sent = []
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
                    sents.append(sent)
                    sent = []
                    content_idx = 0

        print("labeled", len(sents))
        len_counter = Counter()
        for sent in sents:
            len_counter[len(sent)] += 1
        self._bucket_sizes = KMeans(n_bkts, len_counter).splits
        self._buckets = [[] for i in xrange(n_bkts)]
        len2bkt = {}
        prev_size = -1
        for bkt_idx, size in enumerate(self._bucket_sizes):
            len2bkt.update(zip(range(prev_size + 1, size + 1), [bkt_idx] * (size - prev_size)))
            prev_size = size

        self._record = []
        for sent in sents:
            bkt_idx = len2bkt[len(sent)]
            self._buckets[bkt_idx].append(sent)
            idx = len(self._buckets[bkt_idx]) - 1
            self._record.append((bkt_idx, idx))

        for bkt_idx, (bucket, size) in enumerate(zip(self._buckets, self._bucket_sizes)):
            self._buckets[bkt_idx] = np.zeros((size, len(bucket), 6), dtype=np.int32)
            for idx, sent in enumerate(bucket):
                self._buckets[bkt_idx][:len(sent), idx, :] = np.array(sent, dtype=np.int32)

    @property
    def idx_sequence(self):
        return [x[1] for x in sorted(zip(self._record, range(len(self._record))))]

    def get_batches(self, batch_size, shuffle=True):
        batches = []
        for bkt_idx, bucket in enumerate(self._buckets):
            bucket_len = bucket.shape[1]
            n_tokens = bucket_len * self._bucket_sizes[bkt_idx]
            n_splits = max(n_tokens // batch_size, 1)
            range_func = np.random.permutation if shuffle else np.arange
            for bkt_batch in np.array_split(range_func(bucket_len), n_splits):
                batches.append((bkt_idx, bkt_batch))

        if shuffle:
            np.random.shuffle(batches)

        for bkt_idx, bkt_batch in batches:
            word_inputs = self._buckets[bkt_idx][:, bkt_batch, 0]
            lemma_inputs = self._buckets[bkt_idx][:, bkt_batch, 1]
            tag_inputs = self._buckets[bkt_idx][:, bkt_batch, 2]
            arc_targets = self._buckets[bkt_idx][:, bkt_batch, 3]
            rel_targets = self._buckets[bkt_idx][:, bkt_batch, 4]
            pred_targets = self._buckets[bkt_idx][:, bkt_batch, 5]
            yield word_inputs, lemma_inputs, tag_inputs, arc_targets, rel_targets, pred_targets


class AlignmentsLoader(object):
    def __init__(self, input_file):
        self.sents = []
        sent = [0]*150
        with open(input_file) as f:
            print("reading alignment")
            for line in f.readlines():
                info = line.strip().split()

                for word in info:
                    en_index, fr_index = word.split('-')
                    en_index = int(en_index)
                    fr_index = int(fr_index)
                    sent[en_index] = fr_index

                self.sents.append(sent)
                sent = [0] * 150
            print("sens", len(self.sents))


    def get_batches(self, batch_size, shuffle=False):
        batches = []
        sentences_num = len(self.sents)
        batch_num = int(sentences_num*1.0/batch_size)
        batch = []
        print("batch num", batch_num)
        for id, sent in enumerate(self.sents):
            idx = id+1
            if idx > batch_size*batch_num:
                break
            if idx%batch_size==0:
                batch.append(sent)
                batches.append(batch)
                batch = []
            else:
                batch.append(sent)

        for id, batch in enumerate(batches):
            print("alignments shape:", )
            yield batch


class PlainDataLoader(object):
    def __init__(self, input_file, vocab):
        self.sents = []
        sent = []
        with open(input_file) as f:
            print("reading plain text")
            for line in f.readlines():
                info = line.strip().split()
                info.insert(0, "<DUMMY>")
                if info:
                    assert (len(info) > 0), 'Illegal line: %s' % line
                    for word in info:
                        word = vocab.word2id(word.lower())
                        sent.append(word)
                    self.sents.append(sent)
                    sent = []
                else:
                    print(info)
                    print(line)

            print("sens", len(self.sents))


    def get_batches(self, batch_size, shuffle=False):
        batches = []
        sentences_num = len(self.sents)
        batch_num = int(sentences_num*1.0/batch_size)
        batch = []
        lengths = []
        print("batch num", batch_num)
        for id, sent in enumerate(self.sents):
            idx = id+1
            if idx > batch_size*batch_num:
                break
            if idx%batch_size==0:
                batch.append(sent)
                batches.append(batch)
                batch = []
            else:
                batch.append(sent)

        for id, batch in enumerate(batches):
            max_len = 0
            lengths = []
            for sentence in batch:
                lengths.append(len(sentence))
                if len(sentence)>max_len:
                    max_len = len(sentence)
            batch_mask = np.zeros((batch_size, max_len), dtype=np.int32)
            for idx, sent in enumerate(batch):
                batch_mask[idx][:len(sent)] = np.array(sent, dtype=np.int32)

            yield batch_mask.T, lengths