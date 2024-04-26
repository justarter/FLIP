import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import random
import os
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from torch import nn
from torch.cuda.amp import autocast,GradScaler
from sklearn.metrics import roc_auc_score, log_loss
# from dataset import add_special_token
from utils import AvgMeter, get_lr, MetricLogger, SmoothedValue, is_main_process, get_rank
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import json

from mlm_config import create_mlm_parser

class NLP_Model(nn.Module):
    def __init__(self):
        super(NLP_Model, self).__init__()
        self.bert_model = AutoModelForMaskedLM.from_pretrained(cfg.text_encoder_model,
                                                    local_files_only=True)
        self.text_tokenizer_num = len(tokenizer)
        for p in self.bert_model.parameters():
            p.requires_grad = True

    def forward(self, input_ids, attention_mask):
        output = self.bert_model.bert(input_ids=input_ids, attention_mask=attention_mask)
        output = output.last_hidden_state # B N D
        output = self.bert_model.cls(output)

        return output.view(-1, self.text_tokenizer_num)


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
    
def build_loaders(text_input, tokenizer, mode):
    dataset = BertDataset(
        text_input["content"].values,
        text_input["label"].values,
        tokenizer=tokenizer
    )
    sampler = DistributedSampler(dataset, shuffle=True if mode == "train" else False)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        sampler=sampler,
        pin_memory = True,
    )
    return dataloader, sampler

def dynamic_mask(inputs, tokenizer, mask_ratio=0.15, mlm_probability=0.15, special_tokens_mask=None):
    text_id = inputs['input_ids'].clone()
    text_label = inputs['input_ids'].clone()
    probability_matrix = torch.full(text_label.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in text_label.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    text_label[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(text_label.shape, 0.8)).bool() & masked_indices
    text_id[indices_replaced] =  tokenizer.convert_tokens_to_ids( tokenizer.mask_token)
    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(text_label.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len( tokenizer), text_label.shape, dtype=torch.long)
    text_id[indices_random] = random_words[indices_random]
    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    inputs['mask_input_ids'] = text_id
    inputs['mask_text_label'] = text_label
    return inputs

def train_epoch(model, train_loader, optimizer, lr_scheduler, step, loss_fnc):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',  SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss',  SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50  
    logger_data = {} 
    scaler = GradScaler()
    for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
        batch = dynamic_mask(batch, tokenizer)
        ids = batch['mask_input_ids'].to(device, non_blocking=True)
        mask = batch['attention_mask'].to(device, non_blocking=True)
        label = batch['mask_text_label'].to(device, non_blocking=True)
        # accumulate step
        accumulation_steps = cfg.accumulation_steps
        
        output = model(input_ids=ids,attention_mask=mask)
        loss = loss_fnc(output, label.long().view(-1))

        loss = loss / accumulation_steps
        loss.backward()
        if ((i + 1) % accumulation_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()

        logger_data['loss'] = loss.item()
        logger_data['lr'] = optimizer.param_groups[0]["lr"]
        metric_logger.update(logger_data)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    cfg = create_mlm_parser()
    # fix seed
    seed = cfg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    torch.cuda.set_device(cfg.local_rank)
    device = torch.device("cuda", cfg.local_rank)
    dist.init_process_group(backend='nccl')

    data_source = cfg.dataset
    train_text, test_text = make_train_valid_dfs(cfg.struct_path, cfg.text_path, seed, data_source)
    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    
    train_loader, train_sampler = build_loaders(train_text, tokenizer, mode='train')
    test_loader,_ = build_loaders(test_text, tokenizer, mode='test')

    save_path = cfg.output_prefix_path+ str(data_source) + "_mlm.pt"
    print('begin mlm pretrain')
    model = NLP_Model().to(cfg.device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0, end_factor=.0, total_iters=cfg.epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=0)
    
    loss_fn = nn.CrossEntropyLoss()
    step = "epoch"
    print("begain Training")
    
    for epoch in range(cfg.epochs):
        # shuffle
        train_sampler.set_epoch(epoch)
        
        print(f"Epoch: {epoch + 1}")
        print(optimizer.state_dict()['param_groups'][0]['lr'])
        model.train()
        train_stats = train_epoch(model,
                                 train_loader,
                                 optimizer,
                                 lr_scheduler,
                                 step, loss_fn)
        lr_scheduler.step()
        # model.eval()
        # with torch.no_grad():
        #     valid_loss = valid_epoch(model, test_loader, loss_fn,lr_scheduler)

        if is_main_process():    
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                  
            torch.save(model.module, save_path)
            with open(save_path+ "mlm_log.txt","a") as f:
                f.write(json.dumps(log_stats) + "\n")
            print('save model')

        dist.barrier() 
