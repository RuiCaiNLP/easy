# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle

sys.path.append('..')
#import dynet as dy
import numpy as np
from lib import Vocab, DataLoader, PlainDataLoader
from models import AlignParser

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

    plain_data_loader = PlainDataLoader("plain.fr-en.en", vocab)
    plain_data_loader_fr = PlainDataLoader("plain.fr-en.fr", vocab_fr)

    data_loader = DataLoader("processed/train_pro", vocab)
    global_step = 0


    parser = AlignParser(vocab, vocab_fr)
    trainer = optim.Adam(parser.parameters(), lr=0.001)
    """
    for i in parser.mlp_arg_uniScore.parameters():
        i.requires_grad = False
    for i in parser.mlp_pred_uniScore.parameters():
        i.requires_grad = False
    for i in parser.arg_pred_uniScore.parameters():
        i.requires_grad = False
    parser.rel_W.requires_grad = False
    parser.pair_weight.requires_grad = False
    trainer_fr = optim.Adam(filter(lambda p: p.requires_grad, parser.parameters()), lr=0.001)
    for i in parser.mlp_arg_uniScore.parameters():
        i.requires_grad = True
    for i in parser.mlp_pred_uniScore.parameters():
        i.requires_grad = True
    for i in parser.arg_pred_uniScore.parameters():
        i.requires_grad = True

    parser.rel_W.requires_grad = True
    parser.pair_weight.requires_grad = True
    """

    epoch = 0
    best_F1 = 0.
    parser.to(device)

    Plain_English_data_Generator = plain_data_loader.get_batches(batch_size=10)
    Plain_French_data_Generator = plain_data_loader_fr.get_batches(batch_size=10)
    while global_step < 500000:
        print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d' % (epoch,)
        epoch += 1
        for words, tags, preds, rels in \
                data_loader.get_batches(batch_size=10, shuffle=True):
            parser.zero_grad()
            trainer.zero_grad()
            parser.train()

            accuracy, loss = parser("labeled", words, tags, preds, rels, isTrain=True)
            if global_step % 30 == 0:
                print("epoch %d, global step#%d, accuracy:%.2f" %(epoch, global_step, accuracy))
                print(loss)
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parser.parameters(), 5)

            trainer.step()



            if (global_step+1) % 500 == 0:
                with torch.no_grad():
                    print("testing...")
                    parser.eval()
                    correct_noNull_predicts = 0.
                    noNull_predicts = 0.1
                    noNull_labels = 0.0
                    test_data_loader = DataLoader("processed/dev_pro", vocab)
                    for words, tags, preds, rels in \
                            test_data_loader.get_batches(batch_size=10, shuffle=False):
                        a, b, c = parser("labeled", words, tags, preds, rels, isTrain=False)
                        correct_noNull_predicts += a
                        noNull_predicts += b
                        noNull_labels += c

                    P = correct_noNull_predicts/noNull_predicts
                    R = correct_noNull_predicts/noNull_labels
                    F = 2*P*R / (P + R + 0.00001)
                    if F > best_F1:
                        best_F1 = F
                    print(correct_noNull_predicts, noNull_predicts, noNull_labels)
                    print("tested", P, R, F)
                    print("history best:", best_F1)

            # start unlabeled training#
            try:
                words_en, lengths_en = Plain_English_data_Generator.next()
                words_fr, lengths_fr = Plain_French_data_Generator.next()

            except StopIteration:
                print("start a new unlabeled epoch")
                Plain_English_data_Generator = plain_data_loader.get_batches(batch_size=10)
                Plain_French_data_Generator = plain_data_loader_fr.get_batches(batch_size=10)
                words_en, lengths_en = Plain_English_data_Generator.next()
                words_fr, lengths_fr = Plain_French_data_Generator.next()


            # dy.renew_cg()
            """
            parser.zero_grad()
            trainer_fr.zero_grad()
            parser.train()
            loss = parser('unlabeled', (words_en, words_fr))
            if global_step % 30 == 0:
                print("unlabeled: Step #%d:  " %
                      (global_step))
                print(loss)
            for i in parser.mlp_arg_uniScore.parameters():
                i.requires_grad = False
            for i in parser.mlp_pred_uniScore.parameters():
                i.requires_grad = False
            for i in parser.arg_pred_uniScore.parameters():
                i.requires_grad = False

            parser.rel_W.requires_grad = False
            parser.pair_weight.requires_grad = False
            loss.backward()
            trainer_fr.step()

            for i in parser.mlp_arg_uniScore.parameters():
                i.requires_grad = True
            for i in parser.mlp_pred_uniScore.parameters():
                i.requires_grad = True
            for i in parser.arg_pred_uniScore.parameters():
                i.requires_grad = True

            parser.rel_W.requires_grad = True
            parser.pair_weight.requires_grad = True
            """
            global_step += 1






