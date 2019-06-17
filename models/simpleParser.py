# -*- coding: UTF-8 -*-
from __future__ import division
#import dynet as dy
import torch
import numpy as np
import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.nn.init as init
from lib import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class simpleParser(nn.Module):
    def __init__(self, vocab, word_dims=100, pret_dims=100, tag_dims=16,
                 lstm_layers=3, lstm_hiddens=512, dropout_lstm_input=0.3, dropout_lstm_hidden=0.3,
                 mlp_size=500,
                 dropout_mlp=0.2):
        super(simpleParser, self).__init__()
        #pc = dy.ParameterCollection()
        self._vocab = vocab
        #self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
        self.word_embs = nn.Embedding(vocab.words_in_train, word_dims)
        self.word_embs.weight.data.copy_(torch.from_numpy(vocab.get_word_embs(word_dims)))
        #self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs(pret_dims))
        self.pret_word_embs = nn.Embedding(vocab.vocab_size, word_dims)
        self.pret_word_embs.weight.data.copy_(torch.from_numpy(vocab.get_pret_embs(word_dims)))

        self.tag_embs = nn.Embedding(vocab.tag_size, tag_dims)
        self.tag_embs.weight.data.copy_(torch.from_numpy(vocab.get_tag_embs(tag_dims)))

        self.lstm_layers = lstm_layers
        self.lstm_hiddens = lstm_hiddens
        self.dropout_x = dropout_lstm_input
        self.dropout_h = dropout_lstm_hidden
        input_dims = word_dims + pret_dims + tag_dims



        self.BiLSTM = nn.LSTM(input_size=input_dims, hidden_size=lstm_hiddens, batch_first=True,
                                bidirectional=True, num_layers=3)
        #init.orthogonal_(self.BiLSTM.all_weights[0][0])
        #init.orthogonal_(self.BiLSTM.all_weights[0][1])
        #init.orthogonal_(self.BiLSTM.all_weights[1][0])
        #init.orthogonal_(self.BiLSTM.all_weights[1][1])

        self.mlp_arg_uniScore = nn.Sequential(nn.Linear(mlp_size, int(mlp_size/2)),
                                     nn.ReLU(),
                                     nn.Linear(int(mlp_size/2), 1))

        self.mlp_pred_uniScore = nn.Sequential(nn.Linear(mlp_size, int(mlp_size/2)),
                                              nn.ReLU(),
                                              nn.Linear(int(mlp_size/2), 1))

        self.mlp_pred = nn.Linear(2*lstm_hiddens, mlp_size)
        self.mlp_arg = nn.Linear(2 * lstm_hiddens, mlp_size)

        '''
        self.LSTM_builders = []

        self.f_0 = orthonormal_VanillaLSTMBuilder(1, input_dims, lstm_hiddens)
        self.b_0 = orthonormal_VanillaLSTMBuilder(1, input_dims, lstm_hiddens)
        #self.LSTM_builders.append((self.f_0, self.b_0))
        
        self.f_1 = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens)
        self.b_1 = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens)
        self.LSTM_builders.append((self.f_1, self.b_1))
       
        self.f_2 = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens)
        self.b_2 = orthonormal_VanillaLSTMBuilder(1, 2 * lstm_hiddens, lstm_hiddens)
        self.LSTM_builders.append((self.f_2, self.b_2))
'''

        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden
        self.mlp_size = mlp_size


        #orthonormal_initializer(2 * lstm_hiddens, mlp_size)
        #orthonormal_initializer(2 * lstm_hiddens, mlp_size)
        #orthonormal_initializer(2 * lstm_hiddens, mlp_size)
        #W = orthonormal_initializer(2 * lstm_hiddens, mlp_size)
        W = np.random.randn(2*lstm_hiddens, mlp_size).astype(np.float32)
        self.mlp_arg_W = nn.Parameter(torch.from_numpy(W).to(device))
        self.mlp_pred_W = nn.Parameter(torch.from_numpy(W).to(device))
        self.mlp_arg_b = nn.Parameter(torch.from_numpy(np.zeros(mlp_size,).astype("float32")).to(device))
        self.mlp_pred_b = nn.Parameter(torch.from_numpy(np.zeros(mlp_size,).astype("float32")).to(device))
        self.mlp_size = mlp_size
        self.dropout_mlp = dropout_mlp
        self.mlp_dropout = nn.Dropout(p=dropout_mlp)

        self.rel_W = nn.Parameter(torch.from_numpy(np.zeros((mlp_size + 1, vocab.rel_size * (mlp_size + 1))).astype("float32")).to(device))
        #self._pc = pc



    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        # return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
        #        Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))
        return (torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hiddens, requires_grad=False).to(device),
                torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hiddens, requires_grad=False).to(device))


    def forward(self, word_inputs, tag_inputs, pred_golds, rel_targets=None, isTrain=True, given_gold_preds=False):
        # inputs, targets: seq_len x batch_size

        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        marker = self._vocab.PAD
        mask = np.greater(word_inputs, marker).astype(np.int64)
        num_tokens = np.sum(mask, axis=1)
        print(num_tokens)
        print(num_tokens.sum())
        word_embs = self.word_embs(torch.from_numpy(word_inputs.astype('int64')).to(device))
        pre_embs = self.pret_word_embs(torch.from_numpy(word_inputs.astype('int64')).to(device))
        tag_embs = self.tag_embs(torch.from_numpy(tag_inputs.astype('int64')).to(device))


        emb_inputs = torch.cat((word_embs, pre_embs, tag_embs), dim=2)


        init_hidden = self.init_hidden(batch_size)
        top_recur, hidden = self.BiLSTM(emb_inputs, init_hidden)
        del init_hidden

        if isTrain and False:
            dropout_dim = Variable(torch.bernoulli(
                torch.Tensor(1, batch_size, 2*self.lstm_hiddens).fill_(1 - self.dropout_mlp)), requires_grad=False).to(device)
            top_recur = top_recur * dropout_dim.expand(seq_len, batch_size, 2*self.lstm_hiddens)

        #top_recur = top_recur.transpose(0, 1)
        #W_arg, b_arg = self.mlp_arg_W, self.mlp_arg_b
        #W_pred, b_pred = self.mlp_pred_W, self.mlp_pred_b
        # arg_hidden = leaky_relu(dy.affine_transform([b_arg, W_arg, top_recur]))
        g_arg = F.relu(self.mlp_arg(top_recur))
        g_pred = F.relu(self.mlp_pred(top_recur))

        # B T
        uniScores_arg = self.mlp_arg_uniScore(g_arg).view(batch_size, seq_len)
        uniScores_pred = self.mlp_pred_uniScore(g_pred).view(batch_size, seq_len)

        arg_sorted, arg_indices = torch.sort(uniScores_arg, descending=True)
        pred_sorted, pred_indices = torch.sort(uniScores_pred, descending=True)

        preds_num = [len(p) for p in pred_golds]
        candidate_preds_batch = []
        sample_indices_selected = []
        mask_selected = []
        preds_indices_selected = []
        rel_targets_selected = []
        mask_selected = []
        offset_words = 0
        offset_targets = 0

        # labeled data, gold predicates are given
        if isTrain or not isTrain:
            for i, preds in enumerate(pred_golds):
                candidate_preds = preds
                for j in range(preds_num[i]):
                    rel_targets_selected.append(rel_targets[offset_targets+j])
                offset_targets += preds_num[i]
                candidate_preds_batch.append(candidate_preds)

        # only for train, sort it, and then add the top 0.2 portion
        if isTrain:
            for i in range(batch_size):
                candidate_preds_num = int(num_tokens[i]* 0.2)
                sorted_preds = pred_indices[i][: candidate_preds_num].cpu().numpy()
                for candidate in sorted_preds:
                    if not candidate in candidate_preds_batch[i]:
                        candidate_preds_batch[i].append(candidate)
                        null_targets = np.zeros((seq_len), dtype=np.int32)
                        null_targets[:num_tokens[i]] = [42] * num_tokens[i]
                        rel_targets_selected.append(null_targets)

        # find sample indices, mask, preds indices, according to final candidate preds
        for i in range(batch_size):
            for j in range(len(candidate_preds_batch[i])):
                sample_indices_selected.append(i)
                mask_selected.append(mask[i])
                preds_indices_selected.append(candidate_preds_batch[i][j]+offset_words)
            offset_words += int(seq_len)


        g_arg_selected = g_arg.index_select(0, torch.tensor(sample_indices_selected))
        g_pred_selected = g_pred.view(batch_size*seq_len, -1).index_select(0, torch.tensor(preds_indices_selected))
        print(g_pred_selected.size())
        print(g_pred_selected.size())
        print(len(rel_targets_selected))

        W_rel = self.rel_W

        total_preds_num = g_pred_selected.size()[0]


        rel_logits = bilinear(g_arg_selected, W_rel, g_pred_selected, self.mlp_size, seq_len, 1,
                              total_preds_num,
                              num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)

        print(total_preds_num, seq_len)
        flat_rel_logits = rel_logits.view(total_preds_num*seq_len, self._vocab.rel_size)

        if isTrain:

            mask_1D = np.array(mask_selected).reshape(-1)

            rel_preds = torch.argmax(flat_rel_logits, 1).cpu().data.numpy().astype("int64")

            #targets_1D = dynet_flatten_numpy(rel_targets)
            targets_1D = np.array(rel_targets_selected).astype("int64").reshape(-1)

            rel_correct = np.equal(rel_preds, targets_1D).astype(np.float32) * mask_1D
            rel_accuracy = np.sum(rel_correct) / mask_1D.sum()


            loss_function = nn.CrossEntropyLoss(ignore_index=0)

            rel_loss = loss_function(flat_rel_logits, torch.from_numpy(targets_1D).to(device))
            print(rel_accuracy, rel_loss)
            return rel_accuracy, rel_loss
        rel_probs = F.softmax(partial_rel_logits, 1).view(batch_size, seq_len, self._vocab.rel_size).cpu().data.numpy()
        outputs = []

        for msk, pred_gold, rel_prob in zip(np.transpose(mask), pred_golds.T, rel_probs):
            msk[0] = 1.
            sent_len = int(np.sum(msk))
            #rel_prob = rel_prob[np.arange(len(pred_gold)), 0]
            rel_pred = rel_argmax(rel_prob)
            outputs.append(rel_pred[:sent_len])

        return outputs

    def save(self, model, save_path):
        #self._pc.save(save_path)
        torch.save(model.state_dict(), save_path)

    def load(self, load_path):
        self._pc.populate(load_path)