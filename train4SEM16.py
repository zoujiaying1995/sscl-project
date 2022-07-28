# -*- coding: utf-8 -*-
import json
import os
import math
import argparse
import random
import time
from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from criterion import unitedCLLoss
from logger.CSVlogger import CSVlogger
from model import SSCL
from sklearn import metrics
from utils.data_utils import DatesetReader
import numpy as np

from transformers import BertTokenizer, BertConfig, AutoTokenizer


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        stance_dataset = DatesetReader(opt, opt.tokenizer, dataset=opt.dataset)
        self.train_data_loader = DataLoader(dataset=stance_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        self.val_data_loader = DataLoader(dataset=stance_dataset.dev_data, batch_size=opt.batch_size, shuffle=True)
        self.test_data_loader = DataLoader(dataset=stance_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        self.model = opt.model_class(opt).to(opt.device)
        self._print_args()
        self.global_f1 = 0.

        if torch.cuda.is_available():
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for p in self.model.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, unitedCL_criterion, optimizer):
        max_val_acc = 0
        max_val_f1 = 0
        max_val_epoch = 0
        global_step = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = [sample_batched[self.opt.col_map[col]].to(self.opt.device) for col in
                          self.opt.inputs_cols]
                targets = sample_batched[self.opt.col_map['stances']].to(self.opt.device)
                senti_labels = sample_batched[self.opt.col_map['sentiments']].to(self.opt.device)
                features, outputs, fc_input = self.model(inputs)
                cl_loss = torch.tensor(0)
                cl_sentiment = unitedCL_criterion(features, senti_labels)
                cl_stance = unitedCL_criterion(features, targets)
                cl_loss = opt.cl_alpha * cl_sentiment + self.opt.cl_beta * cl_stance
                class_loss = criterion(outputs, targets)
                loss = opt.gama * class_loss + cl_loss
                # print("cl_loss:"+str(cl_loss)+"+ stance_loss:"+str(class_loss))
                loss.backward()
                optimizer.step()
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    print('loss: {:.4f}, class_loss: {:.4f}, cl_loss: {:.4f}, train_acc: {:.4f}'.format(
                        loss.item(), class_loss.item(), cl_loss.item(), train_acc))
            val_acc, val_f1, val_f1_m, _, _, _, _ = self._evaluate_acc_f1(
                self.val_data_loader)
            print(
                'val_acc: {:.4f}, val_f1: {:.4f}, val_f1_m: {:.4f}'.format(
                    val_acc, val_f1, val_f1_m))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1
                max_val_epoch = epoch
                if not os.path.exists('./saved_models/'):
                    os.mkdir('./saved_models/')
                # path = 'state_dict_cl/{0}_{1}{2}'.format(self.opt.model_name, self.opt.dataset, '.pkl')
                torch.save(self.model.state_dict(), opt.save_path)
                print('>> saved: {}'.format(opt.save_path))
            if epoch - max_val_epoch >= self.opt.patience:
                print('>> early stop.')
                break
        return opt.save_path

    def _evaluate_acc_f1(self, data_loader):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[self.opt.col_map[col]].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched[self.opt.col_map['stances']].to(opt.device)
                t_cl_aug, t_outputs, t_fc_input = self.model(t_inputs)
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 2],
                              average='macro')
        f1_mi = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 2],
                                 average='micro')
        f1_against, _, f1_favor = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(),
                                                   average=None)

        f1_0 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0],
                                average='macro')
        f1_1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[1],
                                average='macro')
        f1_2 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[2],
                                average='macro')
        # print('=====================>', f1_against, f1_favor, (f1_against+f1_favor)*0.5)
        f1_m = 0.5 * (f1 + f1_mi)
        f1_all = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2],
                                  average='macro')  # vast
        return test_acc, f1, f1_m, f1_0, f1_1, f1_2, f1_all

    def run(self, repeats=1):
        max_test_acc_avg = 0
        max_test_f1_macro = 0
        max_test_f1_ma_mi_avg = 0
        for i in range(repeats):
            opt.logger.initDF()
            unitedCL_criterion = unitedCLLoss(opt)
            criterion = nn.CrossEntropyLoss()
            self.model = opt.model_class(opt).to(opt.device)
            params = [
                {"params": [value for _, value in self.model.bert.named_parameters() if value.requires_grad],
                 'lr': opt.learning_rate},  # 2e-5
                {"params": [value for _, value in self.model.cl_bert.named_parameters() if value.requires_grad],
                 'lr': opt.learning_rate * 3},  # 6e-5
                {"params": [value for _, value in self.model.fc.named_parameters() if value.requires_grad],
                 'lr': opt.learning_rate * 100},  # 2e-3
                {"params": [value for _, value in self.model.atn.named_parameters() if value.requires_grad],
                 'lr': opt.learning_rate * 100},
                {"params": [value for _, value in self.model.mlp.named_parameters() if value.requires_grad],
                 'lr': opt.learning_rate * 100},
            ]
            # optimizer = self.opt.optimizer(self.model.parameters(), lr=0.00002, weight_decay=1e-5)  #
            optimizer = self.opt.optimizer(params, weight_decay=1e-5)
            print('repeat: ', (i + 1))
            best_model_path = self._train(criterion, unitedCL_criterion, optimizer)
            best_model = self.opt.model_class(opt).to(opt.device)
            best_model.load_state_dict(torch.load(best_model_path))
            test_acc, test_f1, test_f1_m, _, _, _, _ = self._evaluate_acc_f1(
                self.test_data_loader)
            print(
                'test_acc: {:.4f}, test_f1: {:.4f}, test_f1_m: {:.4f}'.format(
                    test_acc, test_f1, test_f1_m))
            # -------------log start-------------------------------#
            log_ = {}
            for arg in vars(self.opt):
                # tmp = '>>> {0}: {1}'.format(arg, getattr(self.opt, arg))
                log_[arg] = getattr(self.opt, arg)
            log_data = {'timestamp': opt.logger.getTimestr(time.time()),
                        'max_test_acc': test_acc, 'max_test_f1_ma_mi_avg': test_f1,
                        'max_test_f1_avg': test_f1_m}
            log_data.update(log_)
            opt.logger.df = opt.logger.df.append(log_data, ignore_index=True)
            if opt.saveLog:
                opt.logger.saveDF()
            # --------------log end------------------------#
            if test_acc > max_test_acc_avg:
                max_test_acc_avg = test_acc
            if test_f1 > max_test_f1_macro:
                max_test_f1_macro = test_f1
            if test_f1_m > max_test_f1_ma_mi_avg:
                max_test_f1_ma_mi_avg = test_f1_m
            print('#' * 100)
        print("max_test_acc_avg:", max_test_acc_avg)
        print("max_test_f1_macro:", max_test_f1_macro)
        print('max_test_f1_ma_mi_avg:', max_test_f1_ma_mi_avg)

        return


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='sentiDCL', type=str)
    parser.add_argument('--dataset', default='toad-mul-hc', type=str,
                        help='toad-mul-hc,toad-mul-fm,toad-mul-la,toad-mul-dt,toad-mul-cc,toad-mul-ath')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_normal_', type=str)
    parser.add_argument('--learning_rate', default=0.00002, type=float)
    parser.add_argument('--l2reg', default=1e-5, type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--saveLog', default=True, type=bool)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument('--bert_path', default='bert-base-uncased', type=str)
    parser.add_argument('--temperature', default=0.13, type=float)  # sentiment_cl
    parser.add_argument('--temperatureP', default=0.13, type=float)  # stance_cl
    parser.add_argument('--cl_alpha', default=0.5, type=float)  # sentiment_cl_weight
    parser.add_argument('--cl_beta', default=0.5, type=float)  # stance_cl_weight
    parser.add_argument('--gama', default=1, type=float)  # stance_cl_weight
    parser.add_argument('--maxlen', default=64, type=int)  # tokenizer maxlen
    parser.add_argument('--patience', default=5, type=int)  # early stop
    parser.add_argument('--mid_layer', default=-1, type=int)  # get bert last layer features
    opt = parser.parse_args()

    model_classes = {
        'sentiDCL': SSCL,
    }
    # targets, texts, stances, sentiments, united_labels, input_idss,attention_masks,token_type_idss
    col_map = {'targets': 0, 'texts': 1, 'stances': 2, 'sentiments': 3, 'united_labels': 4,
               'input_idss': 5, 'attention_masks': 6, 'token_type_idss': 7}

    input_colses = {
        'sentiDCL': ['input_idss', 'attention_masks', 'token_type_idss'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        # 'adadelta': torch.optim.Adadelta,  # default lr=1.0
        # 'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        # 'adamax': torch.optim.Adamax,  # default lr=0.002
        # 'asgd': torch.optim.ASGD,  # default lr=0.01
        # 'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        # 'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    # opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    print("torch.cuda.is_available:" + str(torch.cuda.is_available()))
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.tokenizer = AutoTokenizer.from_pretrained(opt.bert_path)
    opt.bertConfig = BertConfig.from_pretrained(opt.bert_path)
    # opt.bertConfig.attention_probs_dropout_prob = 0.3
    opt.bertConfig.hidden_dropout_prob = 0.15
    opt.bert_hidden_dim = 768  # default=768
    opt.col_map = col_map  # sample_batched[self.opt.col_map[col]] for col in self.opt.inputs_cols
    opt.fc_hidden_dim = 283  # 283
    # settings related to dataset
    prefix = './saved_models/'
    path_ = opt.dataset + '_' + opt.model_name + '.m'
    opt.save_path = os.path.join(prefix, path_)
    log_columns = ['timestamp']
    opt.log_path = './new_log/' + opt.dataset + '_sentiDCL' + '.csv'
    opt.logger = CSVlogger(log_columns, opt.log_path)

    ins = Instructor(opt)
    ins.run(repeats=5)
