import os
import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random

from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from torch import nn
from torch.cuda.amp import autocast,GradScaler
import torch.nn.functional as F
from utils import AvgMeter, get_lr
from sklearn.metrics import roc_auc_score
# from dataset import add_special_token
from sklearn.metrics import log_loss

from mlm_config import create_ctrbert_parser

class Prediction_Layer(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim,dropout):
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.activation(projected)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class CTR_BERT(nn.Module):
    def __init__(self):
        super(CTR_BERT, self).__init__()
        self.user_model = AutoModel.from_pretrained(cfg.text_encoder_model,local_files_only=True)
        self.item_model = AutoModel.from_pretrained(cfg.text_encoder_model,local_files_only=True)

        self.out = Prediction_Layer(2*cfg.text_embedding_dim, 1, 128, dropout=cfg.dropout)
        self.target_token_idx = 0

    def forward(self, user_ids, user_mask, item_ids, item_mask):
        user_out = self.user_model(input_ids=user_ids, attention_mask=user_mask).last_hidden_state # B N D
        item_out = self.item_model(input_ids=item_ids, attention_mask=item_mask).last_hidden_state # B N D
        user_out = user_out[:, self.target_token_idx, :] # B D
        item_out = item_out[:, self.target_token_idx, :] # B D

        output = torch.cat([user_out, item_out], dim=-1) # B 2D
        if cfg.mixed_precision:
            output = self.out(output)
        else:
            output = torch.sigmoid(self.out(output))
        return output



def make_train_valid_dfs(struct_data_path, user_text_data_path, item_text_data_path, seed, data_source):
    struct_data = pd.read_csv(struct_data_path)
    user_text_data = pd.read_table(user_text_data_path, names=["content"], header=None)
    item_text_data = pd.read_table(item_text_data_path, names=["content"], header=None)
    if cfg.sample:
        struct_data,_ = train_test_split(struct_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
        user_text_data,_ = train_test_split(user_text_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
        item_text_data,_ = train_test_split(item_text_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
    # user text data have label
    user_text_data['label'] = struct_data['label']
    train_user_text, test_user_text = user_text_data.iloc[:int(len(user_text_data) * 0.9)].copy(), user_text_data.iloc[int(len(
        user_text_data) * 0.9):].copy()
    train_item_text, test_item_text = item_text_data.iloc[:int(len(item_text_data) * 0.9)].copy(), item_text_data.iloc[int(len(
        item_text_data) * 0.9):].copy()

    return train_user_text, test_user_text, train_item_text, test_item_text

class BertDataset(torch.utils.data.Dataset):
    def __init__(self, user_text_data, item_text_data, text_label, tokenizer):
        self.user_text_data = list(user_text_data)
        self.item_text_data = list(item_text_data)
        self.text_label = text_label
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        user_encoding = self.tokenizer(
            self.user_text_data[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_length//2,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item_encoding = self.tokenizer(
            self.item_text_data[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_length//2,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'user_input_ids': user_encoding['input_ids'].flatten(),
            'user_attention_mask': user_encoding['attention_mask'].flatten(),
            'item_input_ids': item_encoding['input_ids'].flatten(),
            'item_attention_mask': item_encoding['attention_mask'].flatten(),
        }
        item['label'] = torch.tensor(self.text_label[idx], dtype=torch.int)
        return item

    def __len__(self):
        return len(self.user_text_data)

def build_loaders(user_text_input, item_text_input, tokenizer, mode):
    dataset = BertDataset(
        user_text_input["content"].values,
        item_text_input["content"].values,
        user_text_input["label"].values,
        tokenizer=tokenizer
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        shuffle=True if mode == "train" else False
    )
    return dataloader

def train_epoch(model, train_loader, optimizer, step, loss_fnc):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    scaler = GradScaler()
    predicts = []
    labels = []
    for i,batch in enumerate(tqdm_object):
        uids = batch['user_input_ids'].to(cfg.device)
        u_mask = batch['user_attention_mask'].to(cfg.device)
        iids = batch['item_input_ids'].to(cfg.device)
        i_mask = batch['item_attention_mask'].to(cfg.device)
        label = batch['label'].unsqueeze(1).to(cfg.device)
        if cfg.mixed_precision:
            optimizer.zero_grad()
            with autocast():
                output = model(
                    user_ids=uids, user_mask=u_mask,
                    item_ids=iids, item_mask=i_mask)
                loss = loss_fnc(output, label.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            predicts.extend(torch.sigmoid(output).cpu().data.numpy())
            labels.extend(label.cpu().data.numpy())
        else:
            optimizer.zero_grad()
            output = model(
                    user_ids=uids, user_mask=u_mask,
                    item_ids=iids, item_mask=i_mask)
            loss = loss_fnc(output, label.float())
            loss.backward()
            optimizer.step()

            predicts.extend(output.cpu().data.numpy())
            labels.extend(label.cpu().data.numpy())

        count = batch["label"].size(0)
        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))
    
    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
    auc = roc_auc_score(labels, predicts)
    m2 = log_loss(labels, predicts)
    print(f"train auc:{auc}, train logloss:{m2}")
    return loss_meter, auc, m2


def valid_epoch(model, valid_loader, loss_fnc):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(valid_loader, total=len(valid_loader))
    predicts = []
    labels = []
    for batch in tqdm_object:
        uids = batch['user_input_ids'].to(cfg.device)
        u_mask = batch['user_attention_mask'].to(cfg.device)
        iids = batch['item_input_ids'].to(cfg.device)
        i_mask = batch['item_attention_mask'].to(cfg.device)
        label = batch['label'].unsqueeze(1).to(cfg.device)
        output = model(
                user_ids=uids, user_mask=u_mask,
                item_ids=iids, item_mask=i_mask)
        
        loss = loss_fnc(output, label.float())
        count = batch["label"].size(0)
        if cfg.mixed_precision:
            predicts.extend(torch.sigmoid(output).cpu().data.numpy())
        else:
            predicts.extend(output.cpu().data.numpy())
        labels.extend(label.cpu().data.numpy())

        loss_meter.update(loss.item(), count)
        tqdm_object.set_postfix(valid_loss=loss_meter.avg)
    labels = np.concatenate(labels)
    predicts = np.concatenate(predicts)
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
    cfg = create_ctrbert_parser()
    data_source = cfg.dataset
    train_user_text, test_user_text, train_item_text, test_item_text = make_train_valid_dfs(cfg.struct_path, cfg.user_text_path, 
                                                                                            cfg.item_text_path, seed, data_source)
    setup_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    
    train_loader = build_loaders(train_user_text, train_item_text, tokenizer, mode='train')
    test_loader = build_loaders(test_user_text, test_item_text, tokenizer, mode='test')

    save_path = cfg.output_prefix_path+ str(data_source) + "_ctrbert.pt"
    write_path = cfg.output_prefix_path+ str(data_source) + "_ctrbert.txt"
    
    print('begin CTR_BERT')

    model = CTR_BERT()
    model.to(cfg.device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0, end_factor=.0, total_iters=cfg.epochs)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0)
    
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
        writer_text = [epoch, train_auc, train_logloss, valid_auc, valid_logloss, best_auc]

        with open(write_path,'a+') as writer:
            writer.write(' '.join([str(x) for x in writer_text]) + '\n')
