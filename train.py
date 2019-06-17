# -*- coding: UTF-8 -*-
from __future__ import division
import sys, time, os, cPickle

sys.path.append('..')
#import dynet as dy
import numpy as np
from lib import Vocab, DataLoader
from models import simpleParser

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


    vocab = Vocab("processed/train_pro", "glove.6B.100d.txt", 1)
    cPickle.dump(vocab, open("vocab_save", 'w'))

    data_loader = DataLoader("processed/train_pro", vocab)
    global_step = 0

    parser = simpleParser(vocab)
    trainer = optim.Adam(parser.parameters(), lr=0.001)
    epoch = 0
    best_F1 = 0.

    while global_step < 50:
        print time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), '\nStart training epoch #%d' % (epoch,)
        epoch += 1
        for words, tags, preds, rels in \
                data_loader.get_batches(batch_size=10, shuffle=False):
            accuracy, loss = parser(words, tags, preds, rels, isTrain=True, given_gold_preds=False)
            loss.backward()
            trainer.step()


