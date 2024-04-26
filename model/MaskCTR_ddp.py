import os
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from layers.modules import TextEncoder, ProjectionHead, RecEncoder_DeepFM, RecEncoder_DCNv2
from layers.core import align_loss,unif_loss,Max_Sim_2
from preprocessing.inputs import SparseFeat
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
from transformers import AutoModel, AutoTokenizer, BertTokenizer,BertConfig
import torch.distributed as dist
from nce import IndexLinear

class mlm_Prediction_Layer(nn.Module):
    def __init__(self,embedding_dim,output_dim,hidden_dim,dropout):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.activation(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class Simple_Attention(nn.Module):
    def __init__(self, input_size, hidden_size,dropout):
        super().__init__()
        self.linear_s = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        query = self.linear_s(query).unsqueeze(1)
        scores = torch.matmul(query, key.transpose(-2,-1))
        p_attn = torch.softmax(scores,dim=-1)
        out = torch.matmul(p_attn, value).squeeze(1)
        return out

class MaskCTR(nn.Module):
    def __init__(
            self,
            cfg, rec_embedding_dim,
            text_embedding_dim, text_encoder_model,
            pretrained=True, trainable=True,
            struct_linear_feature_columns = None,
            struct_dnn_feature_columns = None,
            struct_feature_num=None,
            text_tokenizer_num=None):
        super(MaskCTR, self).__init__()
        projection_dim = 128

        self.rec_encoder = RecEncoder_DCNv2(struct_linear_feature_columns,struct_dnn_feature_columns, struct_feature_num)
        
        self.text_encoder = TextEncoder(text_encoder_model, pretrained, trainable)
        self.rec_projection = ProjectionHead(rec_embedding_dim, projection_dim, dropout=0.2)
        self.text_projection = ProjectionHead(text_embedding_dim, projection_dim, dropout=0.2)
        # self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True)
        self.temperature = cfg.temperature

        self.struct_field_size = len(struct_dnn_feature_columns)
        self.struct_feature_num = struct_feature_num
        self.text_tokenizer_num = text_tokenizer_num

        self.rec_pred_list = Simple_Attention(rec_embedding_dim, text_embedding_dim, dropout=0.2)

        feat_count = torch.load(cfg.feat_count_path)
        self.config = {'num_fields': self.struct_field_size, 'proj_size':32, 'input_size':struct_feature_num, 'data_path':cfg.data_path,
                       'feat_count':feat_count, 'pt_neg_num':cfg.pt_neg_num, 'pt_loss': cfg.pt_loss}
        self.create_pretrain_predictor(text_embedding_dim + rec_embedding_dim)

        self.text_pred_module = mlm_Prediction_Layer(rec_embedding_dim+text_embedding_dim,
                                                        text_tokenizer_num, hidden_dim=text_embedding_dim, dropout=0.2)
       
        self.rec_dense = nn.Linear(projection_dim,1)
        self.text_dense = nn.Linear(projection_dim,1)

        self.use_mlm = cfg.use_mlm
        self.use_mfm = cfg.use_mfm
        self.use_contrastive = cfg.use_contrastive

    def create_pretrain_predictor(self, input_dim):
        self.feat_encoder = nn.Linear(input_dim, self.config['num_fields'] * self.config['proj_size'])
        self.criterion = IndexLinear(config=self.config)

    def get_pretrain_output(self, input_vec, labels, masked_index):
        '''
        Input:
            input_vec: [batch size, input_vec]

        Return:
            MFP: (loss, signal count, sum of accuracy)
        '''
        batch_size = input_vec.shape[0]
        enc_output = self.feat_encoder(input_vec).view(batch_size, self.config['num_fields'], self.config['proj_size'])
        selected_output = torch.gather(enc_output, 1, masked_index.unsqueeze(-1).repeat(1, 1, self.config['proj_size']))
        loss, G_logits, G_features = self.criterion(labels, selected_output)
        total_acc = (G_logits.argmax(dim=2) == 0).sum().item()
        outputs = (loss, labels.shape[0] * labels.shape[1], total_acc)
        return outputs

    def forward(self,batch):
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        return_list = []

        rec_features = self.rec_encoder(batch["rec_data"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
    
        rec_embeddings = self.rec_projection(rec_features)
        text_embeddings = self.text_projection(text_features)
     
        gathered_rec_embeddings = [torch.zeros_like(rec_embeddings) for _ in range(world_size)]
        gathered_text_embeddings = [torch.zeros_like(text_embeddings) for _ in range(world_size)]
        dist.all_gather(gathered_rec_embeddings, rec_embeddings)
        dist.all_gather(gathered_text_embeddings, text_embeddings)
        all_rec_embeddings = torch.cat(
            [rec_embeddings]
            + gathered_rec_embeddings[:rank]
            + gathered_rec_embeddings[rank+1:]
        )
        all_text_embeddings = torch.cat(
            [text_embeddings]
            + gathered_text_embeddings[:rank]
            + gathered_text_embeddings[rank+1:]
        )

        loss_fn = nn.CrossEntropyLoss()
        logits_per_rec = all_rec_embeddings @ all_text_embeddings.t()/self.temperature
        logits_per_text = logits_per_rec.t()
        ground_truth = torch.arange(len(logits_per_rec)).long()
        ground_truth = ground_truth.to(rec_features.device, non_blocking=True)
        loss_ita = (loss_fn(logits_per_rec, ground_truth) + loss_fn(logits_per_text, ground_truth)) /2

        total_loss = loss_ita
        # print(loss_ita)
        return_list.append(loss_ita)

        if self.use_mfm:
            # nce loss
            mask_rec_features = self.rec_encoder(batch['mask_rec_data'])
            tmp = self.text_encoder.mlm_forward(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            tmp = self.rec_pred_list(mask_rec_features, tmp, tmp)
            rec_pred_in = torch.cat([mask_rec_features, tmp], dim=-1)
            mfm_loss = self.get_pretrain_output(rec_pred_in, batch['mask_rec_label'].long(), batch['mask_rec_index'])[0]

            # print(mfm_loss)
            total_loss = total_loss + mfm_loss
            return_list.append(mfm_loss)

        if self.use_mlm:
            mask_text_features = self.text_encoder.mlm_forward(input_ids=batch["mask_input_ids"], attention_mask=batch["attention_mask"])# B N D
            mask_text_labels = batch['mask_text_label']
          
            text_pred_in = torch.cat([mask_text_features, rec_features.unsqueeze(1).repeat((1,mask_text_features.shape[1],1))], dim=-1) # B N D+H
            text_pred_out = self.text_pred_module(text_pred_in) 
            mlm_loss = nn.CrossEntropyLoss()(text_pred_out.view(-1, self.text_tokenizer_num), mask_text_labels.long().view(-1))
            # print(mlm_loss)
            total_loss = total_loss + mlm_loss
            return_list.append(mlm_loss)

    
        return total_loss, return_list

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()