import pandas as pd
import torch
import random

from sklearn.metrics import log_loss, roc_auc_score
import numpy as np
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing.inputs import SparseFeat, get_feature_names
from model.dcn import DCN
from model.wdm import WideDeep
from model.autoint import AutoInt
from model.deepfm import DeepFM
from model.xdeepfm import xDeepFM
from model.dcnv2 import DCNMix
from model.pnn import PNN
from model.afn import AFN
from model.afm import AFM
from model.base_model import BaseModel

from sklearn.utils import shuffle
from torch import  nn
from utils import check_path
import argparse
import json

# ctr baselines

def data_process(data_path,data_source):
    data = pd.read_csv(data_path,encoding='utf-8')
    train = data.iloc[:int(len(data) * 0.9)].copy()
    test = data.iloc[int(len(data) * 0.9):].copy()
        
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', default=32, type=int, help='embedding dim for CTR model')
    parser.add_argument('--epoch', default=10, type=int, help='training epoch')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--dropout', default=0., type=float, help='dropout rate for DNN in CTR model')
    parser.add_argument('--seed', default=2012, type=int, help='random seed')
    
    parser.add_argument('--model_name', default='DeepFM', type=str, help='CTR model name')
    parser.add_argument('--data_source', default='movielens', type=str, help='dataset name')

    args = parser.parse_args()
    
    embedding_dim = args.embedding_dim
    epoch = args.epoch
    batch_size = args.batch_size
    seed = args.seed
    lr = args.learning_rate
    dropout = args.dropout
    setup_seed(seed)
    
    data_source = args.data_source
    model_name = args.model_name

    if data_source=='movielens':
        data_path = "./data/ml-1m/remap_data.csv"
        meta_path = './data/ml-1m/meta.json'

        sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zipcode','genre', 'title']
        target = ['label']
    elif data_source=='bookcrossing':
        data_path = "./data/BookCrossing/remap_data.csv"
        meta_path = './data/BookCrossing/meta.json'
    
        sparse_features = ['User ID', 'Location', 'Age', 'ISBN', 'Book title', 'Author', 'Publication year', 'Publisher']
        target = ['label']
    elif data_source=='goodreads':
        data_path = "./data/GoodReads/remap_data.csv"
        meta_path = './data/GoodReads/meta.json'
        
        sparse_features = ['User ID','Book ID', 'Book title', 'Book genres' ,'Average rating', 'Number of book reviews', 'Author ID', 'Author name',
                   'Number of pages','eBook flag', 'Format', 'Publisher', 'Publication year', 'Work ID', 'Media type']
        target = ['label']

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

    check_path('baseline_models/')
    check_path('baseline_results/')
    
    es = EarlyStopping(monitor='val_auc', min_delta=0, verbose=1, patience=3, mode='max', baseline=None)
    
    ckpt_path = f'baseline_models/{data_source}_{model_name}_{epoch}_{batch_size}_{lr}_{dropout}.ckpt'
    mdckpt = ModelCheckpoint(filepath=ckpt_path, monitor='val_auc',
                             mode='max', verbose=1, save_best_only=True, save_weights_only=True)

    if model_name == 'DeepFM':
        model = DeepFM(linear_feature_columns, dnn_feature_columns, total_feature_num,  task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'AutoInt':
        model = AutoInt(linear_feature_columns, dnn_feature_columns, total_feature_num, task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'DCN':
        model = DCN(linear_feature_columns, dnn_feature_columns, total_feature_num, task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'WideDeep':
        model = WideDeep(linear_feature_columns, dnn_feature_columns, total_feature_num, task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'DCNv2':
        model = DCNMix(linear_feature_columns, dnn_feature_columns, total_feature_num, task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'xDeepFM':
        model = xDeepFM(linear_feature_columns, dnn_feature_columns, total_feature_num, task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'PNN':
        model = PNN(linear_feature_columns, dnn_feature_columns, total_feature_num,  task='binary', dnn_dropout=dropout,
                        seed=1024,dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'AFN':
        model = AFN(linear_feature_columns, dnn_feature_columns, total_feature_num,  task='binary', dnn_dropout=dropout,
                        seed=1024,afn_dnn_hidden_units=(300,300,128), device=device)
    elif model_name == 'AFM':
        model = AFM(linear_feature_columns, dnn_feature_columns, total_feature_num,  task='binary',
                        seed=1024,device=device)


    model.compile("adam", "binary_crossentropy", metrics=['auc', 'accuracy', 'logloss']
                  , lr=lr)

    test_model_input = {name: test[name] for name in sparse_features }
    
    model.fit(train_model_input, train[target].values, batch_size= batch_size, epochs=epoch, verbose=0, validation_data=[test_model_input, test[target].values],
              callbacks=[es, mdckpt])

    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    test_model_input = {name: test[name] for name in sparse_features  }
    pred_ts = model.predict(test_model_input, batch_size=2048)

    print(args)
    logloss = str(round(log_loss(test[target].values, pred_ts), 4))
    auc = str(round(roc_auc_score(test[target].values, pred_ts), 4))
    print("test LogLoss", logloss)
    print("test AUC", auc)
    writer_text = [str(epoch), str(batch_size), str(lr), str(dropout), auc, logloss]
    with open(f'baseline_results/{data_source}_{model_name}.txt','a+') as writer:
        writer.write(' '.join(writer_text) + '\n')

