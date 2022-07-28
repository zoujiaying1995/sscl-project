# -*- coding: utf-8 -*-

import os
import pickle
import string

import numpy as np
from transformers import BertTokenizer

# stance: {-1:Against, 0:None, 1:Favor}



class Dataset(object):
    def __init__(self, targets, texts, stances, sentiments, united_labels, input_idss, attention_masks,
                 token_type_idss):
        self.targets = targets
        self.texts = texts
        self.stances = stances
        self.input_idss = input_idss
        self.attention_masks = attention_masks
        self.token_type_idss = token_type_idss
        self.sentiments = sentiments
        self.united_labels = united_labels

    def __getitem__(self, index):
        return self.targets[index], self.texts[index], self.stances[index], self.sentiments[index], self.united_labels[
            index], self.input_idss[index], self.attention_masks[index], self.token_type_idss[index]

    def __len__(self):
        return len(self.targets)


sent_map = {'negative': 0, 'positive': 1}


class DatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_raw = lines[i].lower().strip()
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, maxlen):
        print("dataset:" + fname)
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        targets = []
        texts = []
        stances = []
        input_idss = []
        attention_masks = []
        token_type_idss = []
        sentiments = []
        united_labels = []
        unitied_map = {
            "0": {"0": "0", "1": "1", "2": "2"},
            "1": {"0": "3", "1": "4", "2": "5"},
            "2": {"0": "6", "1": "7", "2": "8"},
        }
        for i in range(0, len(lines), 4):
            text = lines[i].lower().strip()
            target = lines[i + 1].lower().strip()
            stance = lines[i + 2].strip()
            sentiment_ = lines[i + 3].strip()

            sentiment = int(sentiment_) + 1
            stance = int(stance)
            if fname.split('/')[-1].find('vast') < 0:
                stance = stance + 1
            united_label_ = unitied_map[str(sentiment)][str(stance)].strip()
            united_label = int(united_label_)
            org_token = tokenizer(
                target, text,
                add_special_tokens=True,
                max_length=maxlen,
                return_tensors='pt',
                padding='max_length',
                truncation=True
            )

            targets.append(target)
            texts.append(text)
            stances.append(stance)
            sentiments.append(sentiment)
            united_labels.append(united_label)
            input_idss.append(org_token['input_ids'])
            attention_masks.append(org_token['attention_mask'])
            token_type_idss.append(org_token['token_type_ids'])

        return targets, texts, stances, sentiments, united_labels, input_idss, attention_masks, token_type_idss

    def __init__(self, opt, tokenizer, dataset='dt_hc'):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'toad-mul-ath': {
                'train': './toad-data/wo_A_train.united',
                'dev': './toad-data/wo_A_val.united',
                'test': './toad-data/A_test.united'
            },
            'toad-mul-dt': {
                'train': './toad-data/wo_DT_train.united',
                'dev': './toad-data/wo_DT_val.united',
                'test': './toad-data/DT_test.united'
            },
            'toad-mul-hc': {
                'train': './toad-data/wo_HC_train.united',
                'dev': './toad-data/wo_HC_val.united',
                'test': './toad-data/HC_test.united'
            },
            'toad-mul-fm': {
                'train': './toad-data/wo_FM_train.united',
                'dev': './toad-data/wo_FM_val.united',
                'test': './toad-data/FM_test.united'
            },
            'toad-mul-la': {
                'train': './toad-data/wo_LA_train.united',
                'dev': './toad-data/wo_LA_val.united',
                'test': './toad-data/LA_test.united'
            },
            'toad-mul-ccrc': {
                'train': './toad-data/wo_CC_train.united',
                'dev': './toad-data/wo_CC_val.united',
                'test': './toad-data/CC_test.united'
            }
            ,
            # covid data ZERO SHOT
            'mul_sc': {  #
                'train': './vast_data/wo_SC.sent',
                'test': './vast_data/SC.sent'
            },
            'mul_af': {  #
                'train': './vast_data/wo_AF.sent',
                'test': './vast_data/AF.sent'
            },
            'mul_wa': {  #
                'train': './vast_data/wo_WA.sent',
                'test': './vast_data/WA.sent'
            },
            'mul_sh': {  #
                'train': './vast_data/wo_SH.sent',
                'test': './vast_data/SH.sent'
            },
            'zeroshot_vast': {
                'train': './vast_data/vast_train.sent',
                'dev': './vast_data/vast_dev.sent',
                'test': './vast_data/vast_test.sent'
            },
            'fewshot_vast': {
                'train': './vast_data/vast_train.sent',
                'dev': './vast_data/few_vast_dev.sent',
                'test': './vast_data/few_vast_test.sent'
            },
        }
        self.tokenizer = tokenizer
        targets, texts, stances, sentiments, united_labels, input_idss, attention_masks, token_type_idss = DatesetReader.__read_data__(
            fname[dataset]['train'], tokenizer, opt.maxlen)
        self.train_data = Dataset(targets, texts, stances, sentiments, united_labels, input_idss, attention_masks,
                                  token_type_idss)
        targets, texts, stances, sentiments, united_labels, input_idss, attention_masks, token_type_idss = DatesetReader.__read_data__(
            fname[dataset]['test'], tokenizer, opt.maxlen)
        self.test_data = Dataset(targets, texts, stances, sentiments, united_labels, input_idss, attention_masks,
                                 token_type_idss)
        try:
            targets, texts, stances, sentiments, united_labels, input_idss, attention_masks, token_type_idss = DatesetReader.__read_data__(
                fname[dataset]['dev'], tokenizer, opt.maxlen)
            self.dev_data = Dataset(targets, texts, stances, sentiments, united_labels, input_idss, attention_masks,
                                    token_type_idss)
        except:
            pass
