import numpy as np
import pandas as pd
import torch
import csv
import random
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from deepctr_torch.callbacks import EarlyStopping, ModelCheckpoint
from preprocessing.inputs import SparseFeat, get_feature_names
from dataset import CtrDataset2
from transformers import AutoModel, AutoTokenizer
from utils import AvgMeter, get_lr, check_path
from tqdm import tqdm
from torch import nn
from layers.core import concat_fun
from finetune_config import create_finetune_all_parser
from preprocessing.inputs import combined_dnn_input
from layers.core import PredictionLayer

# finetune all

class DCNMix_NLP_Model(nn.Module):
    def __init__(self):
        super(DCNMix_NLP_Model, self).__init__()
        self.model = torch.load(load_pretrain_path)
        self.text_dense = nn.Linear(cfg.text_embedding_dim,1)
        self.text_out = PredictionLayer()
        self.out = PredictionLayer()
        # self.alpha = torch.tensor(cfg.alpha)
        self.alpha = torch.nn.Parameter(torch.zeros((1,)), requires_grad=True)
        

    def forward(self, batch):
        sparse_embedding_list, dense_value_list = self.model.rec_encoder.input_from_feature_columns(batch["rec_data"], self.model.rec_encoder.dnn_feature_columns,
                                                                               self.model.rec_encoder.embedding_dict)
    
        logit = self.model.rec_encoder.linear_model(batch["rec_data"])
        dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
        deep_out = self.model.rec_encoder.dnn(dnn_input)
        cross_out = self.model.rec_encoder.crossnet(dnn_input)
        stack_out = torch.cat((cross_out, deep_out), dim=-1)
        logit += self.model.rec_encoder.dnn_linear(stack_out)
        
        text_features = self.model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_logit = self.text_dense(text_features)
        
        if cfg.mixed_precision:
            ctr_pred = logit
            text_pred = text_logit
            y_pred = logit*self.alpha + text_logit*(1-self.alpha)
        else:
            ctr_pred = self.model.rec_encoder.out(logit)
            text_pred = self.text_out(text_logit)
            y_pred = self.out(logit*self.alpha + text_logit*(1-self.alpha))
        return ctr_pred, text_pred, y_pred


def add_regularization_weight(regularization_weight, weight_list, l1=0.0, l2=0.0):
    if isinstance(weight_list, torch.nn.parameter.Parameter):
        weight_list = [weight_list]
    else:
        weight_list = list(weight_list)
    regularization_weight.append((weight_list, l1, l2))
    return regularization_weight

def get_regularization_loss(regularization_weight):
    total_reg_loss = torch.zeros((1,), device='cuda')
    for weight_list, l1, l2 in regularization_weight:
        for w in weight_list:
            if isinstance(w, tuple):
                parameter = w[1]  # named_parameters
            else:
                parameter = w
            if l2 > 0:
                try:
                    total_reg_loss += torch.sum(l2 * torch.square(parameter))
                except AttributeError:
                    total_reg_loss += torch.sum(l2 * parameter * parameter)

    return total_reg_loss

def make_train_valid_dfs(struct_data_path, text_data_path, seed, data_source):
    struct_data = pd.read_csv(struct_data_path)
    text_data = pd.read_table(text_data_path,names = ["content"],header=None)
    if cfg.sample:
        struct_data,_ = train_test_split(struct_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
        text_data,_ = train_test_split(text_data,test_size= (1-cfg.sample_ration) ,random_state= seed)

    text_data['label'] = struct_data['label']
    train_struct, test_struct = struct_data.iloc[:int(len(struct_data) * 0.9)].copy(), struct_data.iloc[int(len(
        struct_data) * 0.9):].copy()
    train_text, test_text = text_data.iloc[:int(len(text_data) * 0.9)].copy(), text_data.iloc[int(len(
        text_data) * 0.9):].copy()

    print('train size, test size: ', len(train_struct),  len(test_struct))
    return train_struct,  test_struct, train_text,  test_text, struct_data


def build_loaders(struct_input, text_input, linear_feature_columns,dnn_feature_columns,tokenizer, mode):
    dataset = CtrDataset2(
        struct_input,
        text_input["content"].values,
        text_input["label"].values,
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        tokenizer=tokenizer,
        max_length=cfg.max_length
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory = True,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def process_struct_data(data_source, train, test, data):
    embedding_dim = 32
    if (data_source == 'movielens'):
        sparse_features = ['user_id', 'gender', 'age', 'occupation',  'zipcode', 'movie_id',  'title','genre']
    elif (data_source == 'bookcrossing'):
        sparse_features = ['User ID', 'Location', 'Age', 'ISBN', 'Book title', 'Author', 'Publication year', 'Publisher']
    elif (data_source == 'goodreads'):
        sparse_features =  ['User ID','Book ID', 'Book title', 'Book genres' ,'Average rating', 'Number of book reviews', 'Author ID', 'Author name',
                   'Number of pages','eBook flag', 'Format', 'Publisher', 'Publication year', 'Work ID', 'Media type']

    sparse_feature_columns = [SparseFeat(feat, data[feat].nunique(), embedding_dim=embedding_dim)
                              for i, feat in enumerate(sparse_features)]
   
    linear_feature_columns = sparse_feature_columns
    dnn_feature_columns = sparse_feature_columns

    train_model_input = {name: train[name] for name in sparse_features }
    test_model_input = {name: test[name] for name in sparse_features  }

    return linear_feature_columns, dnn_feature_columns, train_model_input, test_model_input


def train_epoch(model, train_loader, optimizer, step,loss_fnc):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    scaler = GradScaler()
    predicts = []
    labels = []
    for batch in tqdm_object:
        batch = {k: v.to(cfg.device) for k, v in batch.items() if k != "text_data"}
        label = batch['label'].unsqueeze(1).to(cfg.device)
        
        if cfg.mixed_precision:
            optimizer.zero_grad()
            with autocast():
                ctr_output, text_output, output = model(batch)
                reg_loss = get_regularization_loss(regularization_weight)
                loss = loss_fnc(output, label.float()) + loss_fnc(ctr_output, label.float()) + loss_fnc(text_output, label.float())+ reg_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predicts.extend(torch.sigmoid(output).cpu().data.numpy())
            labels.extend(label.cpu().data.numpy())
        else:
            optimizer.zero_grad()
            ctr_output, text_output, output = model(batch)
            reg_loss = get_regularization_loss(regularization_weight)
       
            loss = loss_fnc(output, label.float()) + loss_fnc(ctr_output, label.float()) + loss_fnc(text_output, label.float()) + reg_loss
            loss.backward()
            optimizer.step()
            predicts.extend(output.cpu().data.numpy())
            labels.extend(label.cpu().data.numpy())

        count = batch["label"].size(0)
        loss_meter.update(loss.item(), count)
        auc = roc_auc_score(label.cpu().data.numpy(), output.cpu().data.numpy())
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    labels = np.concatenate(labels).astype(np.float64)
    predicts = np.concatenate(predicts).astype(np.float64)
    auc = roc_auc_score(labels, predicts)
    m2 = log_loss(labels, predicts)
    print(f"train auc:{auc}, train logloss:{m2}")
    return loss_meter, auc, m2


def valid_epoch(model, valid_loader,loss_fnc):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    predicts = []
    labels = []
    for batch in tqdm_object:
        batch = {k: v.to(cfg.device) for k, v in batch.items() if k != "text_data"}
        label = batch['label'].unsqueeze(1).to(cfg.device)
        ctr_output, text_output, output = model(batch)
        loss = loss_fnc(output, label.float()) + loss_fnc(ctr_output, label.float()) + loss_fnc(text_output, label.float())
        count = batch["label"].size(0)
        if cfg.mixed_precision:
            predicts.extend(torch.sigmoid(output).cpu().data.numpy())
        else:
            predicts.extend(output.cpu().data.numpy())
        labels.extend(label.cpu().data.numpy())
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
        
    labels = np.concatenate(labels).astype(np.float64)
    predicts = np.concatenate(predicts).astype(np.float64)
    auc = roc_auc_score(labels, predicts)
    m2 = log_loss(labels, predicts)
    print(f"valid auc:{auc}, valid logloss:{m2}")
    return loss_meter, auc, m2

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




if __name__ == '__main__':
    seed = 2012
    cfg = create_finetune_all_parser()
    data_source = cfg.dataset
    model = None
    setup_seed(seed)
    train_struct, test_struct, train_text, test_text, struct_data = make_train_valid_dfs(cfg.struct_path,cfg.text_path,seed, data_source)
    linear_feature_columns, dnn_feature_columns, train_struct_input, test_struct_input = \
        process_struct_data(data_source,train_struct,test_struct, struct_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer,local_files_only=True)
    train_loader = build_loaders(train_struct_input, train_text,
                                 linear_feature_columns,dnn_feature_columns,tokenizer, mode='train')
    test_loader = build_loaders(test_struct_input, test_text,
                                linear_feature_columns,dnn_feature_columns,tokenizer, mode ='test')   

    load_pretrain_path = cfg.load_prefix_path + \
                f'Feature_restore_model/{cfg.model_path}/{data_source}{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.pre_epochs}_{cfg.pre_lr}_0.15_0.15_best.pt'
    print('load pretrain path', load_pretrain_path)
    
    save_path = f'Feature_finetune_all_models/{cfg.model_path}/' 
    write_path = f'Feature_finetune_all_results/{cfg.model_path}/'
    check_path(save_path)
    check_path(write_path)
    save_path += f'{data_source}_{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.pre_epochs}_{cfg.pre_lr}_0.15_0.15.pt'
    write_path += f'{data_source}_{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.pre_epochs}_{cfg.pre_lr}_0.15_0.15.txt'
    print('begin finetune all')
    
    if cfg.dataset == 'movielens':
        l2_reg = 1e-4
    elif cfg.dataset == 'bookcrossing':
        l2_reg = 1e-4
    elif cfg.dataset == 'goodreads':
        l2_reg = 0

    model_name = cfg.backbone
    if model_name == 'DCNv2':
        model = DCNMix_NLP_Model()
        
    model.to(cfg.device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    text_param_list, other_param_list = [],[]
    for (name, param) in model.module.named_parameters():
        if 'text_encoder' in name:
            text_param_list.append(param)
        else:
            other_param_list.append(param)

    params = [
        {"params": text_param_list, "lr": 5e-5},
        {"params": other_param_list}
    ]
    optimizer = torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )
 
    print('l2 reg', l2_reg)
    regularization_weight = []
    add_regularization_weight(regularization_weight, model.module.model.rec_encoder.embedding_dict.parameters(), l2=l2_reg)
    add_regularization_weight(regularization_weight, model.module.model.rec_encoder.linear_model.parameters(), l2=l2_reg)
      
    # print(len(regularization_weight))
    
    if cfg.mixed_precision:
        loss_fn = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.BCELoss()
    
    best_auc = -float('inf')
    step = "epoch"
    print("begain Training")

    patience = 3
    es_cnt = 0
    early_stop = False
    for epoch in range(cfg.epochs):
        if early_stop:
            print('early stop')
            break
        print(f"Epoch: {epoch + 1}")
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        model.train()
        train_loss, train_auc, train_logloss = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 step, loss_fn)
        # lr_scheduler.step()

        model.eval()
        with torch.no_grad():
            valid_loss, valid_auc, valid_logloss = valid_epoch(model, test_loader, loss_fn)
            if valid_auc > best_auc:
                best_auc = valid_auc
                es_cnt = 0
                torch.save(model.module, save_path)
                print('save model')
            else:
                es_cnt += 1
            if es_cnt >= patience:
                early_stop = True
        print('train auc {:.5f}, train logloss {:.5f}, valid auc {:.5f}, valid logloss {:.5f}, best auc {:.5f}'.format(
            train_auc, train_logloss, valid_auc, valid_logloss, best_auc))
        
        tmp_alpha = model.module.alpha.cpu().detach().numpy().item()
        print("alpha {:.5f}".format(tmp_alpha))
        writer_text = [epoch, l2_reg, cfg.lr,cfg.batch_size,train_auc, train_logloss, valid_auc, valid_logloss, best_auc, tmp_alpha]

        with open(write_path,'a+') as writer:
            writer.write(' '.join([str(x) for x in writer_text]) + '\n')

