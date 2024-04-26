import torch
import torch.nn as nn

from model.base_model import BaseModel
from layers.core import DNN
from layers.interaction import LogTransformLayer


class AFN(BaseModel):
    def __init__(self,
                 linear_feature_columns, dnn_feature_columns, total_feature_num,
                 ltl_hidden_size=256, afn_dnn_hidden_units=(256, 128),
                 l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0,
                 init_std=0.0001, seed=1024, dnn_dropout=0, dnn_activation='relu',
                 task='binary', device='cpu', gpus=None):

        super(AFN, self).__init__(linear_feature_columns, dnn_feature_columns, total_feature_num, l2_reg_linear=l2_reg_linear,
                                  l2_reg_embedding=l2_reg_embedding, init_std=init_std, seed=seed, task=task,
                                  device=device, gpus=gpus)

        self.ltl = LogTransformLayer(len(dnn_feature_columns), self.embedding_size, ltl_hidden_size)
        self.afn_dnn = DNN(self.embedding_size * ltl_hidden_size, afn_dnn_hidden_units,
                       activation=dnn_activation, l2_reg=l2_reg_dnn, dropout_rate=dnn_dropout, use_bn=True,
                       init_std=init_std, device=device)
        self.afn_dnn_linear = nn.Linear(afn_dnn_hidden_units[-1], 1)
        self.to(device)
    
    def forward(self, X):

        sparse_embedding_list, _ = self.input_from_feature_columns(X, self.dnn_feature_columns,
                                                                   self.embedding_dict)
        logit = self.linear_model(X)
        if len(sparse_embedding_list) == 0:
            raise ValueError('Sparse embeddings not provided. AFN only accepts sparse embeddings as input.')
            
        afn_input = torch.cat(sparse_embedding_list, dim=1)
        ltl_result = self.ltl(afn_input)
        afn_logit = self.afn_dnn(ltl_result)
        afn_logit = self.afn_dnn_linear(afn_logit)
        
        logit += afn_logit
        y_pred = self.out(logit)
        
        return y_pred