from transformers import BertModel
import torch.nn as nn
import torch
from torch.nn import functional as F


class ATN(nn.Module):
    def __init__(self, opt) -> None:
        super(ATN, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.QW = nn.Linear(opt.bert_hidden_dim, opt.bert_hidden_dim)
        self.KW = nn.Linear(opt.bert_hidden_dim, opt.bert_hidden_dim)
        self.VW = nn.Linear(opt.bert_hidden_dim, opt.bert_hidden_dim)

    def forward(self, q, k, v):
        ks = self.KW(k)  # Batch * N * opt.hidden_dim
        qs = self.QW(q)  # Batch * N * opt.hidden_dim
        sims = torch.bmm(qs, ks.permute(0, 2, 1))  # Batch * N * N
        sims = torch.softmax(sims, dim=-1)

        vs = self.VW(v)
        vs = torch.bmm(sims, vs)

        return vs


class SSCL(nn.Module):
    def __init__(self, opt):
        super(SSCL, self).__init__()
        self.opt = opt
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=opt.bertConfig)
        self.cl_bert = BertModel.from_pretrained('bert-base-uncased', config=opt.bertConfig)
        self.fc = nn.Sequential(nn.Linear(opt.bert_hidden_dim, opt.fc_hidden_dim), nn.Dropout(0.2),
                                nn.Linear(opt.fc_hidden_dim, opt.polarities_dim))
        self.dropout = nn.Dropout(0.2)
        self.mlp = nn.Linear(opt.bert_hidden_dim, opt.bert_hidden_dim)
        self.atn = ATN(opt)

    def forward(self, inputs):
        input_ids, attention_mask, token_type_ids = inputs  # 32 * 1 * 64
        input_ids = input_ids.reshape(input_ids.shape[0], -1)  # shape [batch_size,max_len]
        attention_mask = attention_mask.reshape(attention_mask.shape[0], -1)
        token_type_ids = token_type_ids.reshape(token_type_ids.shape[0], -1)
        cl_input_ids = torch.cat([input_ids, input_ids], dim=0)  # shape [batch_size*2,max_len]
        cl_attention_mask = torch.cat([attention_mask, attention_mask], dim=0)
        cl_token_type_ids = torch.cat([token_type_ids, token_type_ids], dim=0)
        # # get target-specific features
        hidden_feats, cls_feats, output_hidden_states = self.bert(input_ids, attention_mask=attention_mask,
                                                                  token_type_ids=token_type_ids,
                                                                  output_hidden_states=True, return_dict=False)

        # output_hidden_states[mid_layer][:,0,:] = shape[batch_size,1,768]
        hidden_feats1, cls_feats1, output_hidden_states1 = self.cl_bert(cl_input_ids,
                                                                        attention_mask=cl_attention_mask,
                                                                        token_type_ids=cl_token_type_ids,
                                                                        output_hidden_states=True,
                                                                        return_dict=False)  # for contrastive learning
        # Feature Fusion
        mid_layer_feats = output_hidden_states1[self.opt.mid_layer]  # batch*2,seq,hidden_dim
        batch_size_ = int(mid_layer_feats.shape[0] / 2)
        mid_layer_feats = mid_layer_feats[:batch_size_, ::]
        fc_input = self.atn(mid_layer_feats, hidden_feats, hidden_feats)  # batch,seq,hidden_dim
        fc_input = torch.mean(fc_input, dim=1).squeeze(1)
        fc_input = F.relu(fc_input)
        fc_input = self.dropout(fc_input)
        output = self.fc(fc_input)

        cls_feats1 = self.mlp(cls_feats1)  # for contrastive learning
        features = cls_feats1.unsqueeze(1)  # batch_size*2,1,64
        features = F.normalize(features, dim=2)
        features = F.relu(features)
        return features, output, fc_input
