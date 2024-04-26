import pandas as pd
import torch
import random
from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing.inputs import SparseFeat, get_feature_names
from sklearn.utils import shuffle
from model.base_model import BaseModel
from layers.core import concat_fun
from torch import nn
import argparse
from utils import str2bool, check_path
import json
from preprocessing.inputs import combined_dnn_input
import os

# finetune ctr

def data_process(data_path, data_source):
    data = pd.read_csv(data_path, encoding='utf-8')
    train = data.iloc[:int(len(data)*0.9)].copy()
    test = data.iloc[int(len(data)*0.9):].copy()
    return train, test, data

def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class DCNMix(BaseModel):
    def __init__(self, linear_feature_columns,
                 dnn_feature_columns, total_feature_num,  cross_num=2,
                 dnn_hidden_units=(300,300, 128), l2_reg_linear=0.00001,
                 l2_reg_embedding=0.00001, l2_reg_cross=0.00001, l2_reg_dnn=0, init_std=0.0001, seed=1024,
                 dnn_dropout=0, low_rank=32, num_experts=4,
                 dnn_activation='relu', dnn_use_bn=False, task='binary', device='cpu', gpus=None):

        super(DCNMix, self).__init__(linear_feature_columns=linear_feature_columns,
                                     dnn_feature_columns=dnn_feature_columns, total_feature_num= total_feature_num, l2_reg_embedding=l2_reg_embedding,
                                     init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)
        
        self.ctr_model = torch.load(load_pretrain_path+"rec_best.pt")
        self.rec_projection = torch.load(load_pretrain_path+"rec_projection.pt")
        if use_rec_projection_head:
            self.dnn_linear = nn.Linear(projection_dim, 1, bias=False)
        self.regularization_weight = []
        self.add_regularization_weight(self.ctr_model.embedding_dict.parameters(), l2=l2_reg_embedding)
        self.add_regularization_weight(self.ctr_model.linear_model.parameters(), l2=l2_reg_linear)
        self.to(device)

    def forward(self, inputs):
        sparse_embedding_list, dense_value_list = self.ctr_model.input_from_feature_columns(inputs, self.ctr_model.dnn_feature_columns,
                                                                               self.ctr_model.embedding_dict)
    
        logit = self.ctr_model.linear_model(inputs)
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)

        deep_out = self.ctr_model.dnn(dnn_input)
        cross_out = self.ctr_model.crossnet(dnn_input)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)
        logit += self.ctr_model.dnn_linear(stack_out)
    
        y_pred = self.ctr_model.out(logit)
        return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=32, type=int, help='embedding dim for CTR model')
    parser.add_argument('--epoch', default=20, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate for DNN in CTR model')
    parser.add_argument('--seed', default=2012, type=int, help='random seed')
    parser.add_argument('--weight_decay', default=0.001, type=float)
    parser.add_argument('--l2_reg', default=0.001, type=float)
    parser.add_argument("--model_path", type = str) 
    parser.add_argument('--model_name', default='DCNv2',type=str, help='CTR model name') 
    parser.add_argument('--data_source', type=str, help='dataset name') # movielens, bookcrossing, goodreads
    parser.add_argument("--use_rec_head", type = str2bool, default=False, help = "use projection head")
    parser.add_argument("--use_rec_dense", type = str2bool, default=False, help = "")
    
    parser.add_argument("--post_fix", type = str, default='', help = "")
    
    parser.add_argument("--obs", type = str2bool, default=True, help = "")  
    parser.add_argument("--gpu", type = int, default=0, help = "")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

    embedding_dim = args.embedding_dim
    epoch = args.epoch
    batch_size = args.batch_size
    seed = args.seed
    lr = args.learning_rate
    dropout = args.dropout
    use_rec_projection_head = args.use_rec_head
    use_rec_dense = args.use_rec_dense
    setup_seed(seed)

    data_source = args.data_source
    model_name = args.model_name
    projection_dim = 128
    
    args.load_prefix_path = "./"
    args.output_prefix_path = './'
        
    if data_source=='movielens':
        data_path = args.load_prefix_path + "data/ml-1m/remap_data.csv"
        meta_path = args.load_prefix_path + 'data/ml-1m/meta.json'

        sparse_features =  ['user_id', 'gender', 'age', 'occupation',  'zipcode', 'movie_id',  'title','genre']
        target = ['label']
        l2_reg = 0. #1e-3
        if model_name == 'DCNv2':
            tmp_path = 'DCNv2_ML_30'

    elif data_source=='bookcrossing':
        data_path = args.load_prefix_path + "./data/BookCrossing/remap_data.csv"
        meta_path = args.load_prefix_path + './data/BookCrossing/meta.json'
        
        sparse_features = ['User ID', 'Location', 'Age', 'ISBN', 'Book title', 'Author', 'Publication year', 'Publisher']
        target = ['label']
        l2_reg = 0 #1e-3
    
        if model_name == 'DCNv2':
            tmp_path = 'DCNv2_BX_30'

    elif data_source=='goodreads':
        data_path = args.load_prefix_path + "data/GoodReads/remap_data.csv"
        meta_path = args.load_prefix_path + 'data/GoodReads/meta.json'
        
        sparse_features = ['User ID','Book ID', 'Book title', 'Book genres' ,'Average rating', 'Number of book reviews', 'Author ID', 'Author name',
                   'Number of pages','eBook flag', 'Format', 'Publisher', 'Publication year', 'Work ID', 'Media type']
        target = ['label']
        l2_reg = 0
      
        if model_name == 'DCNv2':
            tmp_path = 'DCNv2_GD_10'


    tmp_path = args.model_path
    l2_reg = args.l2_reg
    print('l2 reg', l2_reg)

    load_pretrain_path = args.load_prefix_path + f'Feature_restore_model/{tmp_path}/{data_source}{args.post_fix}_'
    print('load path: ',load_pretrain_path)

    ckpt_path = f'Feature_finetune_ctr_models/{tmp_path}/'
    write_path = f'Feature_finetune_ctr_results/{tmp_path}/'
    check_path(ckpt_path)
    check_path(write_path)
    ckpt_path += f'{data_source}_{model_name}_{epoch}_{batch_size}_{lr}_{dropout}_{use_rec_projection_head}_{args.post_fix}.ckpt'
    write_path += f'{data_source}_{model_name}_{args.post_fix}.txt'
    
    train,test,data = data_process(data_path,data_source)

    with open(meta_path) as fh:
        meta_data = json.load(fh)
    total_feature_num = meta_data['feature_num']
    print("total feature num: ", total_feature_num)

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                            for i, feat in enumerate(sparse_features)]
    
    linear_feature_columns = sparse_feature_columns
    dnn_feature_columns = sparse_feature_columns
    train_model_input = {name: train[name] for name in sparse_features}

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = "cuda"

    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1,
                       patience=3, mode='max', baseline=None)

    mdckpt = ModelCheckpoint(filepath=ckpt_path, monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    if model_name == 'DCNv2':
        model = DCNMix(linear_feature_columns, dnn_feature_columns,total_feature_num, task='binary', dnn_dropout=dropout,
                        l2_reg_linear=l2_reg, l2_reg_embedding=l2_reg, device=device)
        
    weight_decay = args.weight_decay
    print('weight decay', weight_decay)

    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)

    test_model_input = {name: test[name] for name in sparse_features }
    model.fit(train_model_input, train[target].values, batch_size= batch_size, epochs=epoch, verbose=0, validation_data=[test_model_input, test[target].values],
              callbacks=[es, mdckpt])

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    test_model_input = {name: test[name] for name in sparse_features }
    pred_ts = model.predict(test_model_input, batch_size=2048)

    print(args)
    logloss = str(round(log_loss(test[target].values, pred_ts), 4))
    auc = str(round(roc_auc_score(test[target].values, pred_ts), 4))
    print("test LogLoss", logloss)
    print("test AUC", auc)
    writer_text = [str(epoch),str(l2_reg), str(weight_decay), str(batch_size), str(lr), str(dropout), str(use_rec_projection_head), auc, logloss]
    with open(write_path,'a+') as writer:
        writer.write(' '.join(writer_text) + '\n')
