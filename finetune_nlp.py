import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torch import nn
from torch.cuda.amp import autocast,GradScaler
from utils import AvgMeter, get_lr, check_path
from sklearn.metrics import roc_auc_score, log_loss
from finetune_config import create_finetune_nlp_parser

# finetune nlp

class NLP_Model(nn.Module):
    def __init__(self):
        super(NLP_Model, self).__init__()
        self.bert_model = torch.load(load_pretrain_path + "text_best.pt")
        self.out = nn.Linear(cfg.text_embedding_dim, 1)
        # self.dropout = nn.Dropout(0.5) # roberta-large ML 0.2

    def forward(self, input_ids, attention_mask):
        output = self.bert_model(input_ids, attention_mask)
        # output = self.dropout(output)

        if cfg.mixed_precision:
            output = self.out(output)
        else:
            output = torch.sigmoid(self.out(output))
        return output

def make_train_valid_dfs(struct_data_path, text_data_path, seed, data_source):
    struct_data = pd.read_csv(struct_data_path)
    text_data = pd.read_table(text_data_path, names=["content"], header=None)
    if cfg.sample:
        struct_data,_ = train_test_split(struct_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
        text_data,_ = train_test_split(text_data,test_size= (1-cfg.sample_ration) ,random_state= seed)

    text_data['label'] = struct_data['label']
    train_text, test_text = text_data.iloc[:int(len(text_data) * 0.9)].copy(), text_data.iloc[int(len(
        text_data) * 0.9):].copy()

    return train_text, test_text


def build_loaders(text_input, tokenizer, mode):
    dataset = BertDataset(
        text_input["content"].values,
        text_input["label"].values,
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


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, text_data, text_label, tokenizer):
        self.text_data = list(text_data)
        self.text_label = text_label
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.text_data[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=cfg.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        item['label'] = torch.tensor(self.text_label[idx], dtype=torch.int)
        return item

    def __len__(self):
        return len(self.text_data)

def train_epoch(model, train_loader, optimizer, step, loss_fnc):
    loss_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    scaler = GradScaler()
    predicts = []
    labels = []

    for i,batch in enumerate(tqdm_object):
        ids = batch['input_ids'].to(cfg.device)
        mask = batch['attention_mask'].to(cfg.device)
        label = batch['label'].unsqueeze(1).to(cfg.device)
        if cfg.mixed_precision:
            optimizer.zero_grad()
            with autocast():
                output = model(input_ids=ids, attention_mask=mask)
                loss = loss_fnc(output, label.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            predicts.extend(torch.sigmoid(output).cpu().data.numpy())
            labels.extend(label.cpu().data.numpy())
        else:
            optimizer.zero_grad()
            output = model(input_ids=ids, attention_mask=mask)
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
        ids = batch['input_ids'].to(cfg.device)
        mask = batch['attention_mask'].to(cfg.device)
        label = batch['label'].unsqueeze(1).to(cfg.device)
        
        output = model(input_ids=ids, attention_mask=mask)
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
    cfg = create_finetune_nlp_parser()
    data_source = cfg.dataset
    train_text, test_text = make_train_valid_dfs(cfg.struct_path, cfg.text_path, seed, data_source)
    setup_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    
    train_loader = build_loaders(train_text, tokenizer, mode='train')
    test_loader = build_loaders(test_text, tokenizer, mode='test')
    
    load_pretrain_path = cfg.load_prefix_path + \
                f'Feature_restore_model/{cfg.model_path}/{data_source}{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.pre_epochs}_{cfg.pre_lr}_0.15_0.15_'
    
    save_path = f'Feature_finetune_nlp_models/{cfg.model_path}/' 
    write_path =  f'Feature_finetune_nlp_results/{cfg.model_path}/'
    check_path(save_path)
    check_path(write_path)
    save_path += f'{data_source}_{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.pre_epochs}_{cfg.pre_lr}_0.15_0.15.pt'
    write_path += f'{data_source}_{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.pre_epochs}_{cfg.pre_lr}_0.15_0.15.txt'
    print('begin finetune nlp')
 
    model = NLP_Model()
    
    model.to(cfg.device)
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    
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
        writer_text = [epoch, cfg.lr,cfg.batch_size,train_auc, train_logloss, valid_auc, valid_logloss, best_auc]

        with open(write_path,'a+') as writer:
            writer.write(' '.join([str(x) for x in writer_text]) + '\n')

