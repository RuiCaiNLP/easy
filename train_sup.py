# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle

sys.path.append('..')
#import dynet as dy
import numpy as np
from lib import Vocab, DataLoader, PlainDataLoader
from models import AlignSup

import argparse

import torch
import torch.tensor
import torch.autograd
from torch.autograd import Variable
from torch.nn.utils import clip_grad_value_
from torch import optim
import time
import random
import copy



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    np.random.seed(666)
    torch.manual_seed(1)

    print("loading English vocab...")
    vocab = Vocab("processed/train_pro", "glove.6B.100d.txt", 1)

    print("loading French vocab...")
    vocab_fr = Vocab("processed/train_pro_fr", "less.fr.300.vec", 1)

    plain_data_loader_fr = PlainDataLoader("translate_fr", vocab_fr)

    data_loader = DataLoader("processed/train_pro", vocab)
    global_step = 0
    parser = AlignSup(vocab, vocab_fr)
    trainer = optim.Adam(parser.parameters(), lr=0.001)
    epoch = 0
    best_F1 = 0.
    best_F1_fr = 0.
    parser.to(device)
    while global_step < 500000:
        print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d' % (epoch,)
        epoch += 1
        for s1, s2 in \
                zip(data_loader.get_batches(batch_size=5, shuffle=False, clip=True), plain_data_loader_fr.get_batches(batch_size=5)):
            words, tags, preds, rels = s1
            words_fr, _ = s2
            parser.zero_grad()
            trainer.zero_grad()
            parser.train()

            accuracy, loss, a1, l1, a2, l2, a3, l3 = parser("labeled", (words, words_fr), tags, preds, rels,  isTrain=True)
            if global_step % 1 == 0:
                print("epoch %d, global step#%d, accuracy:%.2f" %(epoch, global_step, accuracy))
                print(loss)
                print("epoch %d, global step#%d, accuracy:%.2f" % (epoch, global_step, a1))
                print(l1, l2, l3)
            loss += l1 + l2 + l3
            loss.backward()
            trainer.step()



            if (global_step+1) % 134 == 0:
                with torch.no_grad():
                    print("testing...")
                    parser.eval()
                    correct_noNull_predicts = 0.
                    noNull_predicts = 0.1
                    noNull_labels = 0.0
                    test_data_loader = DataLoader("processed/dev_pro", vocab)
                    for words, tags, preds, rels in \
                            test_data_loader.get_batches(batch_size=5, shuffle=False):
                        a, b, c = parser("English", words, tags, preds, rels, isTrain=False, clip=False)
                        correct_noNull_predicts += a
                        noNull_predicts += b
                        noNull_labels += c

                    P = correct_noNull_predicts/noNull_predicts
                    R = correct_noNull_predicts/noNull_labels
                    F = 2*P*R / (P + R + 0.00001)
                    if F > best_F1:
                        best_F1 = F
                    print(correct_noNull_predicts, noNull_predicts, noNull_labels)
                    print("English tested", P, R, F)
                    print("English history best:", best_F1)
                with torch.no_grad():
                    print("testing...")
                    parser.eval()
                    correct_noNull_predicts = 0.
                    noNull_predicts = 0.1
                    noNull_labels = 0.0
                    test_data_loader = DataLoader("processed/dev_pro_fr", vocab_fr)
                    for words, tags, preds, rels in \
                            test_data_loader.get_batches(batch_size=5, shuffle=False, clip=False):
                        a, b, c = parser("French", words, tags, preds, rels, isTrain=False)
                        correct_noNull_predicts += a
                        noNull_predicts += b
                        noNull_labels += c

                    P = correct_noNull_predicts / noNull_predicts
                    R = correct_noNull_predicts / noNull_labels
                    F = 2 * P * R / (P + R + 0.00001)
                    if F > best_F1:
                        best_F1_fr = F
                    print(correct_noNull_predicts, noNull_predicts, noNull_labels)
                    print("French tested", P, R, F)
                    print("English history best:", best_F1_fr)
            global_step += 1






