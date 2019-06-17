# -*- coding: UTF-8 -*-
#import dynet as dy
import torch
import numpy as np
import torch.autograd
import torch.nn as nn
import numpy as np
from data import Vocab

import numpy as np
import torch
import math
import torch.nn as nn
import torch.autograd
from torch.autograd import Variable
import torch.nn.init as init
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.nn.init as init
from numpy import random as nr
from operator import itemgetter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def orthonormal_VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens):
   # builder = dy.VanillaLSTMBuilder(lstm_layers, input_dims, lstm_hiddens)
    builder = nn.LSTMCell(input_size=input_dims, hidden_size=lstm_hiddens).to(device)
    """ 
    for layer, params in enumerate(builder.get_parameters()):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer > 0 else input_dims))
        W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
        params[0].set_value(np.concatenate([W_x] * 4, 0))
        params[1].set_value(np.concatenate([W_h] * 4, 0))
        b = np.zeros(4 * lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens:2 * lstm_hiddens] = -1.0
        params[2].set_value(b)
        
            for layer in range(lstm_layers):
        W = orthonormal_initializer(lstm_hiddens, lstm_hiddens + (lstm_hiddens if layer > 0 else input_dims))
        W_h, W_x = W[:, :lstm_hiddens], W[:, lstm_hiddens:]
        builder.weight_ih.data = torch.from_numpy(np.concatenate([W_x] * 4)).to(device)
        builder.weight_hh.data = torch.from_numpy(np.concatenate([W_h] * 4)).to(device)
        b = np.zeros(4 * lstm_hiddens, dtype=np.float32)
        b[lstm_hiddens:2 * lstm_hiddens] = -1.0
        builder.bias_ih.data = torch.from_numpy(b).to(device)
        builder.bias_hh.data = torch.from_numpy(b).to(device)
         """
    #init.orthogonal_(builder.weight_hh)
    #init.orthogonal_(builder.weight_ih)


    return builder


def biLSTM(builders, inputs, batch_size=None, dropout_x=0., dropout_h=0.):
    seq_len = len(inputs)
    for f, b in builders:
        #print(f,b)
        #f, b = fb.initial_state(), bb.initial_state()
        #fb.set_dropouts(dropout_x, dropout_h)
        #bb.set_dropouts(dropout_x, dropout_h)
        #if batch_size is not None:
        #    fb.set_dropout_masks(batch_size)
        #    bb.set_dropout_masks(batch_size)
        #fs, bs = f.transduce(inputs), b.transduce(reversed(inputs))
        fs = []
        bs = []
        #r_inputs = inputs[::-1]
        hx = torch.zeros((batch_size, 400), requires_grad=False).to(device)
        cx = torch.zeros((batch_size, 400), requires_grad=False).to(device)
        r_hx = torch.zeros((batch_size, 400), requires_grad=False).to(device)
        r_cx = torch.zeros((batch_size, 400), requires_grad=False).to(device)
        input_size = inputs[0].size()[1]
        dropout_mask_x = Variable(torch.bernoulli(
            torch.Tensor(batch_size, input_size).fill_(1 - dropout_x)), requires_grad=False).to(device)
        dropout_mask_h = Variable(torch.bernoulli(
            torch.Tensor(batch_size, 400).fill_(1 - dropout_h)), requires_grad=False).to(device)

        dropout_mask_r_x = Variable(torch.bernoulli(
            torch.Tensor(batch_size, input_size).fill_(1 - dropout_x)),requires_grad=False).to(device)
        dropout_mask_r_h = Variable(torch.bernoulli(
            torch.Tensor(batch_size, 400).fill_(1 - dropout_h)),requires_grad=False).to(device)
        for i in range(len(inputs)):
            #hx *= dropout_mask_h
            inputs_dropped = inputs[i] #* dropout_mask_x
            hx, cx = f(inputs_dropped, (hx, cx))
            fs.append(hx)

            #r_hx *= dropout_mask_r_h
            r_inputs_dropped = inputs[seq_len-1-i] #* dropout_mask_r_x
            r_hx, r_cx = b(r_inputs_dropped, (r_hx, r_cx))
            bs.append(r_hx)
        #bs = bs[::-1]
        #inputs = [torch.cat((f, b), 1) for f, b in zip(fs, bs)]
        outputs = []
        for i in range(len(inputs)):
            forward = fs[i]
            backward = bs[seq_len-1-i]
            bi_h = torch.cat((forward, backward), 1)
            outputs.append(bi_h)
        inputs = outputs
    return inputs


def leaky_relu(x):
    #return dy.bmax(.1 * x, x)
    return torch.max(.1 * x, x)


# def bilinear(x, W, y, input_size, seq_len, batch_size, num_outputs = 1, bias_x = False, bias_y = False):
# 	# x,y: (input_size x seq_len) x batch_size
# 	if bias_x:
# 		x = dy.concatenate([x, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
# 	if bias_y:
# 		y = dy.concatenate([y, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

# 	nx, ny = input_size + bias_x, input_size + bias_y
# 	# W: (num_outputs x ny) x nx
# 	lin = W * x
# 	if num_outputs > 1:
# 		lin = dy.reshape(lin, (ny, num_outputs * seq_len), batch_size = batch_size)
# 	blin = dy.transpose(y) * lin
# 	if num_outputs > 1:
# 		blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size = batch_size)
# 	# seq_len_y x seq_len_x if output_size == 1
# 	# seq_len_y x num_outputs x seq_len_x else
# 	return blin

def bilinear_(x, W, y, input_size, x_seq_len, y_seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    # x,y: input_size x seq_len
    if bias_x:
        x = dy.concatenate([x, dy.inputTensor(np.ones((1, x_seq_len), dtype=np.float32))])
    if bias_y:
        y = dy.concatenate([y, dy.inputTensor(np.ones((1, y_seq_len), dtype=np.float32))])

    nx, ny = input_size + bias_x, input_size + bias_y
    # W: (num_outputs x ny) x nx
    lin = W * x
    if num_outputs > 1:
        lin = dy.reshape(lin, (ny, num_outputs * x_seq_len), batch_size=batch_size)
    blin = dy.transpose(y) * lin
    if num_outputs > 1:
        blin = dy.reshape(blin, (y_seq_len, num_outputs, x_seq_len), batch_size=batch_size)
    # seq_len_y x seq_len_x if output_size == 1
    # seq_len_y x num_outputs x seq_len_x else
    return blin

def bilinear(x, W, y, input_size, x_seq_len, y_seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
    if bias_x:
        bias_one = torch.ones((batch_size, x_seq_len, 1)).to(device)
        x = torch.cat((x, Variable(bias_one)), 2)
    if bias_y:
        bias_one = torch.ones((batch_size, 1)).to(device)
        y = torch.cat((y, Variable(bias_one)), 1)
    nx, ny = input_size + bias_x, input_size + bias_y

    left_part = torch.mm(x.view(batch_size * x_seq_len, -1), W)
    left_part = left_part.view(batch_size, x_seq_len * num_outputs, -1)
    y = y.view(batch_size, -1, 1)
    tag_space = torch.bmm(left_part, y).view(
        batch_size, x_seq_len, num_outputs)

    return tag_space

def orthonormal_initializer(output_size, input_size):
    """
	adopted from Timothy Dozat https://github.com/tdozat/Parser/blob/master/lib/linalg.py
	"""
    print (output_size, input_size)
    I = np.eye(output_size)
    lr = .1
    eps = .05 / (output_size + input_size)
    success = False
    tries = 0
    while not success and tries < 10:
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
        for i in xrange(100):
            QTQmI = Q.T.dot(Q) - I
            loss = np.sum(QTQmI ** 2 / 2)
            Q2 = Q ** 2
            Q -= lr * Q.dot(QTQmI) / (
            np.abs(Q2 + Q2.sum(axis=0, keepdims=True) + Q2.sum(axis=1, keepdims=True) - 1) + eps)
            if np.max(Q) > 1e6 or loss > 1e6 or not np.isfinite(loss):
                tries += 1
                lr /= 2
                break
        success = True
    if success:
        print('Orthogonal pretrainer loss: %.2e' % loss)
    else:
        print('Orthogonal pretrainer failed, using non-orthogonal random matrix')
        Q = np.random.randn(input_size, output_size) / np.sqrt(output_size)
    return np.transpose(Q.astype(np.float32))


def rel_argmax(rel_probs):
    rel_probs[:, Vocab.PAD] = 0
    rel_preds = np.argmax(rel_probs, axis=1)
    return rel_preds
