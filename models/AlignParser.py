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

class AlignParser(nn.Module):
    def __init__(self, vocab, vocab_fr, word_dims=100, pret_dims=100, tag_dims=16,
                 lstm_layers=3, lstm_hiddens=200, dropout_lstm_input=0.5, dropout_lstm_hidden=0.3,
                 mlp_size=150, dropout_mlp=0.2,
                 word_dims_fr=300, pret_dims_fr=300, tag_dims_fr=16,
                 lstm_layers_fr=3, lstm_hiddens_fr=200, dropout_lstm_input_fr=0.5, dropout_lstm_hidden_fr=0.3,
                 mlp_size_fr=150, dropout_mlp_fr=0.2
                 ):
        super(AlignParser, self).__init__()
        #pc = dy.ParameterCollection()
        self._vocab = vocab
        #self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
        self.word_embs = nn.Embedding(vocab.words_in_train, word_dims)
        self.word_embs.weight.data.copy_(torch.from_numpy(vocab.get_word_embs(word_dims)))
        #self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs(pret_dims))
        self.pret_word_embs = nn.Embedding(vocab.vocab_size, word_dims)
        self.pret_word_embs.weight.data.copy_(torch.from_numpy(vocab.get_pret_embs(word_dims)))

        self.lstm_layers = lstm_layers
        self.lstm_hiddens = lstm_hiddens
        self.dropout_x = dropout_lstm_input
        self.dropout_h = dropout_lstm_hidden
        input_dims = word_dims + pret_dims

        self.BiLSTM = nn.LSTM(input_size=input_dims, hidden_size=lstm_hiddens, batch_first=True,
                                bidirectional=True, num_layers=3)
        init.orthogonal_(self.BiLSTM.all_weights[0][0])
        init.orthogonal_(self.BiLSTM.all_weights[0][1])
        init.orthogonal_(self.BiLSTM.all_weights[1][0])
        init.orthogonal_(self.BiLSTM.all_weights[1][1])

        self.mlp_arg_uniScore = nn.Sequential(nn.Linear(2*lstm_hiddens, 150),
                                              nn.ReLU(),
                                              nn.Dropout(0.2),
                                              nn.Linear(150, 150),
                                              nn.ReLU(),
                                              nn.Dropout(0.2),
                                              nn.Linear(150, 1))

        self.mlp_pred_uniScore = nn.Sequential(nn.Linear(2*lstm_hiddens, 150),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                                nn.Linear(150, 150),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Linear(150, 1))

        self.arg_pred_uniScore = nn.Sequential(nn.Linear(4 * lstm_hiddens, 150),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Linear(150, 150),
                                               nn.ReLU(),
                                               nn.Dropout(0.2),
                                               nn.Linear(150, 1))

        self.mlp_pred = nn.Linear(2*lstm_hiddens, mlp_size)
        self.mlp_arg = nn.Linear(2 * lstm_hiddens, mlp_size)
        self.dropout_lstm_input = dropout_lstm_input
        self.dropout_lstm_hidden = dropout_lstm_hidden
        self.mlp_size = mlp_size

        W = np.random.randn(2*lstm_hiddens, mlp_size).astype(np.float32)
        self.mlp_arg_W = nn.Parameter(torch.from_numpy(W).to(device))
        self.mlp_pred_W = nn.Parameter(torch.from_numpy(W).to(device))
        self.mlp_arg_b = nn.Parameter(torch.from_numpy(np.zeros(mlp_size,).astype("float32")).to(device))
        self.mlp_pred_b = nn.Parameter(torch.from_numpy(np.zeros(mlp_size,).astype("float32")).to(device))
        self.mlp_size = mlp_size
        self.dropout_mlp = dropout_mlp
        self.emb_dropout = nn.Dropout(p=dropout_lstm_input)
        self.hidden_dropout = nn.Dropout(p=dropout_lstm_hidden)
        self.mlp_dropout = nn.Dropout(p=dropout_mlp)

        self.rel_W = nn.Parameter(torch.from_numpy(np.zeros((2*lstm_hiddens+1, vocab.rel_size * (2*lstm_hiddens+1))).astype("float32")).to(device))
        #self._pc = pc
        self.pair_weight = nn.Parameter(torch.Tensor([0.5, 0.5]))

        self._vocab_fr = vocab_fr
        # self.word_embs = pc.lookup_parameters_from_numpy(vocab.get_word_embs(word_dims))
        self.word_embs_fr = nn.Embedding(vocab_fr.words_in_train, word_dims_fr)
        self.word_embs_fr.weight.data.copy_(torch.from_numpy(vocab_fr.get_word_embs(word_dims_fr)))
        # self.pret_word_embs = pc.lookup_parameters_from_numpy(vocab.get_pret_embs(pret_dims))
        self.pret_word_embs_fr = nn.Embedding(vocab_fr.vocab_size, word_dims_fr)
        self.pret_word_embs_fr.weight.data.copy_(torch.from_numpy(vocab_fr.get_pret_embs(word_dims_fr)))

        self.lstm_layers_fr = lstm_layers_fr
        self.lstm_hiddens_fr = lstm_hiddens_fr
        self.dropout_x_fr = dropout_lstm_input_fr
        self.dropout_h_fr = dropout_lstm_hidden_fr
        input_dims_fr = word_dims_fr + pret_dims_fr

        self.BiLSTM_fr = nn.LSTM(input_size=input_dims_fr, hidden_size=lstm_hiddens, batch_first=True,
                              bidirectional=True, num_layers=3)
        init.orthogonal_(self.BiLSTM_fr.all_weights[0][0])
        init.orthogonal_(self.BiLSTM_fr.all_weights[0][1])
        init.orthogonal_(self.BiLSTM_fr.all_weights[1][0])
        init.orthogonal_(self.BiLSTM_fr.all_weights[1][1])

        self.hidden_dropout_fr = nn.Dropout(p=dropout_lstm_hidden)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hiddens, requires_grad=False).to(device),
                torch.zeros(self.lstm_layers * 2, batch_size, self.lstm_hiddens, requires_grad=False).to(device))


    def forward(self, pattern, word_inputs, tag_inputs=None, pred_golds=None, rel_targets=None, isTrain=True):
        # inputs, targets: seq_len x batch_size
        if pattern == "unlabeled":
            loss = self.semi_forward(word_inputs)
            return loss
        batch_size = word_inputs.shape[0]
        seq_len = word_inputs.shape[1]
        marker = self._vocab.PAD
        mask = np.greater(word_inputs, marker).astype(np.int64)

        num_tokens = np.sum(mask, axis=1)

        word_embs = self.word_embs(torch.from_numpy(word_inputs.astype('int64')).to(device))
        pre_embs = self.pret_word_embs(torch.from_numpy(word_inputs.astype('int64')).to(device))


        emb_inputs = torch.cat((word_embs, pre_embs), dim=2)
        emb_inputs = self.emb_dropout(emb_inputs)

        init_hidden = self.init_hidden(batch_size)
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(emb_inputs, num_tokens)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, last_hidden = self.BiLSTM(embeds_sort, init_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        top_recur = hidden_states[unsort_idx]
        top_recur = self.hidden_dropout(top_recur)
        del init_hidden

        #g_arg = F.relu(self.mlp_arg(top_recur))
        #g_pred = F.relu(self.mlp_pred(top_recur))
        g_arg = top_recur
        g_pred = top_recur

        # B T
        uniScores_arg = self.mlp_arg_uniScore(g_arg).view(batch_size, seq_len)
        uniScores_pred = self.mlp_pred_uniScore(g_pred).view(batch_size, seq_len)

        arg_sorted, arg_indices = torch.sort(uniScores_arg, descending=True)
        pred_sorted, pred_indices = torch.sort(uniScores_pred, descending=True)

        preds_num = [len(p) for p in pred_golds]
        candidate_preds_batch = []
        sample_indices_selected = []
        preds_indices_selected = []
        rel_targets_selected = []
        mask_selected = []
        offset_words = 0
        offset_targets = 0

        # labeled data, gold predicates are given
        if isTrain or not isTrain:
            for i, preds in enumerate(pred_golds):
                candidate_preds = [ele for ele in preds]
                #for j in range(preds_num[i]):
                #   rel_targets_selected.append(rel_targets[offset_targets+j])

                candidate_preds_batch.append(candidate_preds)



        # only for train, sort it, and then add the top 0.2 portion
        if isTrain:
            for i in range(batch_size):
                candidate_preds_num = int(num_tokens[i]* 0.4)
                sorted_preds = pred_indices[i][: candidate_preds_num].cpu().numpy()
                for candidate in sorted_preds:
                    if not candidate in candidate_preds_batch[i]:
                        candidate_preds_batch[i].append(candidate)


        for i, candidate_preds in enumerate(candidate_preds_batch):
            for j, candidate in enumerate(candidate_preds):
                if j < preds_num[i]:
                    rel_targets_selected.append(rel_targets[offset_targets + j])

                else:
                    null_targets = np.zeros((seq_len), dtype=np.int32)
                    null_targets[:num_tokens[i]] = [42] * num_tokens[i]
                    rel_targets_selected.append(null_targets)
            offset_targets += preds_num[i]
        # find sample indices, mask, preds indices, according to final candidate preds
        for i in range(batch_size):
            for j in range(len(candidate_preds_batch[i])):
                sample_indices_selected.append(i)
                mask_selected.append(mask[i])
                preds_indices_selected.append(candidate_preds_batch[i][j]+offset_words)
            offset_words += int(seq_len)

        g_arg_selected = g_arg.contiguous().index_select(0, torch.tensor(sample_indices_selected).to(device))
        g_pred_selected = g_pred.contiguous().view(batch_size*seq_len, -1).index_select(0, torch.tensor(preds_indices_selected).to(device))

        #print(sample_indices_selected)
        #print(candidate_preds_batch)
        #print(rel_targets_selected)

        W_rel = self.rel_W

        total_preds_num = g_pred_selected.size()[0]


        bilinear_scores = bilinear(g_arg_selected, W_rel, g_pred_selected, self.mlp_size, seq_len, 1,
                              total_preds_num,
                              num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)

        g_pred_selected_expand = g_pred_selected.view(total_preds_num, 1, -1).expand(-1, seq_len, -1)

        g_pred_arg_pair = torch.cat((g_arg_selected, g_pred_selected_expand), 2)
        g_pair_UniScores = self.arg_pred_uniScore(g_pred_arg_pair)

        pair_weight = F.softmax(self.pair_weight, dim=0)

        biaffine_scores = pair_weight[0]*bilinear_scores + pair_weight[1]*g_pair_UniScores

        uniScores_pred = uniScores_pred.view(batch_size, seq_len, 1)
        uniScores_pred_selected = \
            uniScores_pred.view(batch_size*seq_len, 1).index_select(0, torch.tensor(preds_indices_selected).to(device))
        uniScores_pred_selected = uniScores_pred_selected.view(total_preds_num, 1, 1).expand(-1, seq_len, self._vocab.rel_size)

        uniScores_arg = uniScores_arg.view(batch_size, seq_len, 1).expand(-1, -1, self._vocab.rel_size)
        uniScores_arg_selected = uniScores_arg.index_select(0, torch.tensor(sample_indices_selected).to(device))

        rel_logits = biaffine_scores + uniScores_arg_selected  + uniScores_pred_selected

        ##enforce the score of null to be 0
        rel_logits[:, :, 42] = torch.zeros(total_preds_num, seq_len, requires_grad=False).to(device)
        flat_rel_logits = rel_logits.view(total_preds_num*seq_len, self._vocab.rel_size)[:, 1:]



        #print(rel_targets_selected)
        if isTrain:

            mask_1D = np.array(mask_selected).reshape(-1)

            rel_preds = torch.argmax(flat_rel_logits, 1).cpu().data.numpy().astype("int64")

            #targets_1D = dynet_flatten_numpy(rel_targets)
            targets_1D = np.array(rel_targets_selected).astype("int64").reshape(-1)

            rel_correct = np.equal(rel_preds, targets_1D-1).astype(np.float32) * mask_1D
            rel_accuracy = np.sum(rel_correct) / mask_1D.sum()

            loss_function = nn.CrossEntropyLoss(ignore_index=-1)

            targets_1D = (torch.from_numpy(targets_1D)-1).to(device)

            rel_loss = loss_function(flat_rel_logits, targets_1D)

            return rel_accuracy, rel_loss


        ##test
        rel_probs = F.softmax(flat_rel_logits, 1).view(total_preds_num, seq_len, (self._vocab.rel_size-1)).cpu().data.numpy()
        rel_predicts = np.argmax(rel_probs, 2)
        correct_noNull_predict = 0
        noNull_predict = 0
        noNull_labels = 0

        show = 0
        for msk, label_gold, label_predict in zip(mask_selected, rel_targets, rel_predicts):
            if show == 0:
                #print("++++++++++++++++++++++++++++++++++++++++++++++")
                #print(label_gold)
                #print(label_predict)
                show = 1
            for i in range(len(label_predict)):
                if msk[i] > 0:
                    if label_gold[i]-1 != 41:
                        noNull_labels += 1
                    if label_predict[i] != 41:
                        noNull_predict+= 1
                        if label_predict[i] == label_gold[i]-1:
                            correct_noNull_predict += 1

        return correct_noNull_predict, noNull_predict, noNull_labels

    def save(self, model, save_path):
        #self._pc.save(save_path)
        torch.save(model.state_dict(), save_path)

    def load(self, load_path):
        self._pc.populate(load_path)

    def semi_forward(self, word_inputs):
        word_inputs_en, word_inputs_fr = word_inputs
        batch_size = word_inputs_en.shape[0]
        seq_len_en = word_inputs_en.shape[1]
        seq_len_fr = word_inputs_fr.shape[1]
        marker = self._vocab.PAD
        mask_en = np.greater(word_inputs_en, marker).astype(np.int64)
        mask_fr = np.greater(word_inputs_fr, marker).astype(np.int64)

        num_tokens_en = np.sum(mask_en, axis=1)
        num_tokens_fr = np.sum(mask_fr, axis=1)

        word_embs_en = self.word_embs(torch.from_numpy(word_inputs_en.astype('int64')).to(device))
        pre_embs_en = self.pret_word_embs(torch.from_numpy(word_inputs_en.astype('int64')).to(device))

        emb_inputs_en = torch.cat((word_embs_en, pre_embs_en), dim=2)
        emb_inputs_en = self.emb_dropout(emb_inputs_en)

        init_hidden = self.init_hidden(batch_size)
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(emb_inputs_en, num_tokens_en)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, last_hidden = self.BiLSTM(embeds_sort, init_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        top_recur = hidden_states[unsort_idx]
        top_recur = self.hidden_dropout(top_recur)

        g_arg = top_recur
        g_pred = top_recur

        # B T
        uniScores_arg = self.mlp_arg_uniScore(g_arg).view(batch_size, seq_len_en)
        uniScores_pred = self.mlp_pred_uniScore(g_pred).view(batch_size, seq_len_en)

        pred_sorted, pred_indices = torch.sort(uniScores_pred, descending=True)

        candidate_preds_batch = []
        sample_indices_selected = []
        preds_indices_selected = []
        rel_targets_selected = []
        mask_selected = []
        offset_words = 0
        offset_targets = 0

        # labeled data, gold predicates are given

        for i in range(batch_size):
            candidate_preds = []
            candidate_preds_batch.append(candidate_preds)

        # only for train, sort it, and then add the top 0.2 portion

        for i in range(batch_size):
            candidate_preds_num = int(num_tokens_en[i] * 0.4)
            sorted_preds = pred_indices[i][: candidate_preds_num].cpu().numpy()
            for candidate in sorted_preds:
                if not candidate in candidate_preds_batch[i]:
                    candidate_preds_batch[i].append(candidate)

        # find sample indices, mask, preds indices, according to final candidate preds
        for i in range(batch_size):
            for j in range(len(candidate_preds_batch[i])):
                sample_indices_selected.append(i)
                mask_selected.append(mask_en[i])
                preds_indices_selected.append(candidate_preds_batch[i][j] + offset_words)
            offset_words += int(seq_len_en)

        g_arg_selected = g_arg.contiguous().index_select(0, torch.tensor(sample_indices_selected).to(device))
        g_pred_selected = g_pred.contiguous().view(batch_size * seq_len_en, -1).index_select(0, torch.tensor(
            preds_indices_selected).to(device))

        # print(sample_indices_selected)
        # print(candidate_preds_batch)
        # print(rel_targets_selected)

        W_rel = self.rel_W

        total_preds_num = g_pred_selected.size()[0]

        bilinear_scores = bilinear(g_arg_selected, W_rel, g_pred_selected, self.mlp_size, seq_len_en, 1,
                                   total_preds_num,
                                   num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)

        g_pred_selected_expand = g_pred_selected.view(total_preds_num, 1, -1).expand(-1, seq_len_en, -1)

        g_pred_arg_pair = torch.cat((g_arg_selected, g_pred_selected_expand), 2)
        g_pair_UniScores = self.arg_pred_uniScore(g_pred_arg_pair)

        pair_weight = F.softmax(self.pair_weight, dim=0)

        biaffine_scores = pair_weight[0] * bilinear_scores + pair_weight[1] * g_pair_UniScores

        uniScores_pred_selected = \
            uniScores_pred.view(batch_size * seq_len_en, 1).index_select(0,
                                                                      torch.tensor(preds_indices_selected).to(device))
        uniScores_pred_selected = uniScores_pred_selected.view(total_preds_num, 1, 1).expand(-1, seq_len_en,
                                                                                             self._vocab.rel_size)

        uniScores_arg = uniScores_arg.view(batch_size, seq_len_en, 1).expand(-1, -1, self._vocab.rel_size)
        uniScores_arg_selected = uniScores_arg.index_select(0, torch.tensor(sample_indices_selected).to(device))

        rel_logits = biaffine_scores + uniScores_arg_selected + uniScores_pred_selected

        ##enforce the score of null to be 0
        rel_logits[:, :, 42] = torch.zeros(total_preds_num, seq_len_en, requires_grad=False).to(device)
        flat_rel_logits = rel_logits.view(total_preds_num * seq_len_en, self._vocab.rel_size)[:, 1:]

        rel_probs = F.softmax(flat_rel_logits, 1).view(total_preds_num, seq_len_en,
                                                       (self._vocab.rel_size - 1)).cpu().data.numpy()
        rel_predicts = np.argmax(rel_probs, 2)



        """
        for i in range(total_preds_num):
            print(preds_indices_selected[i])
            predicate = word_inputs_en.reshape(batch_size*seq_len_en,)[preds_indices_selected[i]]
            print(predicate)
            print(self._vocab.id2word(predicate))
            roles = list(rel_predicts[i]+1)
            print(self._vocab.id2rel(roles))
            """

        word_embs_fr = self.word_embs_fr(torch.from_numpy(word_inputs_fr.astype('int64')).to(device))
        pre_embs_fr = self.pret_word_embs_fr(torch.from_numpy(word_inputs_fr.astype('int64')).to(device))

        emb_inputs_fr = torch.cat((word_embs_fr, pre_embs_fr), dim=2)
        emb_inputs_fr = self.emb_dropout(emb_inputs_fr)

        init_hidden = self.init_hidden(batch_size)
        embeds_sort, lengths_sort, unsort_idx = self.sort_batch(emb_inputs_fr, num_tokens_fr)
        embeds_sort = rnn.pack_padded_sequence(embeds_sort, lengths_sort, batch_first=True)
        # hidden states [time_steps * batch_size * hidden_units]
        hidden_states, last_hidden = self.BiLSTM_fr(embeds_sort, init_hidden)
        # it seems that hidden states is already batch first, we don't need swap the dims
        # hidden_states = hidden_states.permute(1, 2, 0).contiguous().view(self.batch_size, -1, )
        hidden_states, lens = rnn.pad_packed_sequence(hidden_states, batch_first=True)
        # hidden_states = hidden_states.transpose(0, 1)
        top_recur_fr = hidden_states[unsort_idx]
        top_recur_fr = self.hidden_dropout_fr(top_recur)
        del init_hidden

        # g_arg = F.relu(self.mlp_arg(top_recur))
        # g_pred = F.relu(self.mlp_pred(top_recur))
        g_arg_fr = top_recur_fr
        g_pred_fr = top_recur_fr



        top_recur_fr_T = top_recur_fr.transpose(1, 2)
        atten_matrix = torch.bmm(top_recur, top_recur_fr_T)

        atten_e2f = F.softmax(atten_matrix, dim=2)

        weighted_fr = torch.bmm(atten_e2f, top_recur_fr)

        uniScores_arg_cp = self.mlp_arg_uniScore(weighted_fr).view(batch_size, seq_len_en)
        uniScores_pred_cp = self.mlp_pred_uniScore(weighted_fr).view(batch_size, seq_len_en)

        ## immitate the argument hidden states
        g_arg_selected_cp = weighted_fr.contiguous().index_select(0, torch.tensor(sample_indices_selected).to(device))
        g_pred_selected_cp = weighted_fr.contiguous().view(batch_size * seq_len_en, -1).index_select(0, torch.tensor(
            preds_indices_selected).to(device))


        bilinear_scores_cp = bilinear(g_arg_selected_cp, W_rel, g_pred_selected_cp , self.mlp_size, seq_len_en, 1,
                                   total_preds_num,
                                   num_outputs=self._vocab.rel_size, bias_x=True, bias_y=True)

        g_pred_selected_expand_cp = g_pred_selected_cp.view(total_preds_num, 1, -1).expand(-1, seq_len_en, -1)

        g_pred_arg_pair_cp = torch.cat((g_arg_selected_cp, g_pred_selected_expand_cp), 2)
        g_pair_UniScores_cp = self.arg_pred_uniScore(g_pred_arg_pair_cp)

        pair_weight_cp = F.softmax(self.pair_weight, dim=0)

        biaffine_scores_cp = pair_weight_cp[0] * bilinear_scores_cp + pair_weight_cp[1] * g_pair_UniScores_cp

        uniScores_pred_selected_cp = \
            uniScores_pred_cp.view(batch_size * seq_len_en, 1).index_select(0,
                                                                         torch.tensor(preds_indices_selected).to(
                                                                             device))
        uniScores_pred_selected_cp = uniScores_pred_selected_cp.view(total_preds_num, 1, 1).expand(-1, seq_len_en,
                                                                                             self._vocab.rel_size)

        uniScores_arg_cp = uniScores_arg_cp.view(batch_size, seq_len_en, 1).expand(-1, -1, self._vocab.rel_size)
        uniScores_arg_selected_cp = uniScores_arg_cp.index_select(0, torch.tensor(sample_indices_selected).to(device))

        rel_logits_cp = biaffine_scores_cp + uniScores_arg_selected_cp + uniScores_pred_selected_cp

        ##enforce the score of null to be 0
        rel_logits_cp[:, :, 42] = torch.zeros(total_preds_num, seq_len_en, requires_grad=False).to(device)
        flat_rel_logits_cp = rel_logits_cp.view(total_preds_num * seq_len_en, self._vocab.rel_size)[:, 1:]



        unlabeled_loss_function = nn.KLDivLoss(reduce=False)
        teacher_softmax = F.softmax(flat_rel_logits, dim=1).detach()
        student_softmax = F.log_softmax(flat_rel_logits_cp, dim=1)
        loss = unlabeled_loss_function(student_softmax, teacher_softmax)
        loss = torch.sum(loss)
        return loss


    @staticmethod
    def sort_batch(x, l):
        l = torch.from_numpy(np.asarray(l))
        l_sorted, sidx = l.sort(0, descending=True)
        x_sorted = x[sidx]
        _, unsort_idx = sidx.sort()
        return x_sorted, l_sorted, unsort_idx
