import warnings 
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pretrain_config import create_parser
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score
from preprocessing.inputs import SparseFeat, get_feature_names
from torch.cuda.amp import autocast,GradScaler
from dataset import CtrDataset3
import random
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from model.MaskCTR_ddp import MaskCTR
from utils import AvgMeter, get_lr, MetricLogger, SmoothedValue, is_main_process, get_rank
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import time
import datetime
import json

def make_train_valid_dfs(struct_data_path, text_data_path, seed, data_source):
    struct_data = pd.read_csv(struct_data_path)
    text_data = pd.read_table(text_data_path,names = ["content"],header=None)
    if cfg.sample:
        struct_data,_ = train_test_split(struct_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
        text_data,_ = train_test_split(text_data,test_size= (1-cfg.sample_ration) ,random_state= seed)
    
    text_data['label'] = struct_data['label']
    train_struct, test_struct = struct_data.iloc[:int(len(struct_data) * 0.9)].copy(),\
                                    struct_data.iloc[int(len(struct_data) * 0.9):].copy()
    train_text, test_text = text_data.iloc[:int(len(text_data) * 0.9)].copy(), text_data.iloc[int(len(
        text_data) * 0.9):].copy()
    
    return train_struct,test_struct,train_text,test_text, struct_data

def process_struct_data(data_source, train, test, data):
    embedding_dim = 32
    if (data_source == 'movielens'):
        sparse_features = ['user_id', 'gender', 'age', 'occupation', 'zipcode','movie_id', 'title','genre']
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

def build_loaders(struct_input, text_input, linear_feature_columns,dnn_feature_columns,tokenizer, mode):
    dataset = CtrDataset3(
        struct_input,
        text_input["content"].values,
        text_input["label"].values,
        linear_feature_columns=linear_feature_columns,
        dnn_feature_columns=dnn_feature_columns,
        tokenizer=tokenizer,
        max_length=cfg.max_length,
        mask_ratio=cfg.mask_ratio
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

def dynamic_mask(inputs, mask_ratio, same_column):
    batch_size = inputs['rec_data'].shape[0]
    num_field = inputs['rec_data'].shape[1]
    if same_column: # mask same column
        mask_index = inputs['mask_text_index']
    else:
        mask_num = int(num_field * mask_ratio) 
        mask_index = torch.rand((batch_size, num_field), device=inputs['rec_data'].device)
        mask_index = torch.argsort(mask_index, dim=-1)[:, :mask_num]
    inputs['mask_rec_index'] = mask_index
    inputs['mask_rec_label'] = torch.gather(inputs['rec_data'], 1, mask_index)
    inputs['mask_rec_data'] = torch.scatter(inputs['rec_data'], 1, mask_index, 0)
    return inputs

def train_epoch(model, train_loader, optimizer, lr_scheduler, step):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr',  SmoothedValue(window_size=50, fmt='{value:.6f}'))
    for i in range(loss_num):
        metric_logger.add_meter(f'loss_{i}',  SmoothedValue(window_size=50, fmt='{value:.4f}'))
    
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50  
    logger_data = {} 
    scaler = GradScaler()
    if cfg.mixed_precision:
        for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            batch = dynamic_mask(batch, cfg.mask_ratio, cfg.same_column)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            # loss_ita, loss_mfm, loss_mlm = model(batch)
            with autocast():
                loss, loss_list = model(batch)
            assert len(loss_list) == loss_num

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
         
            if step == "batch":
                lr_scheduler.step()
            for i in range(loss_num):
                logger_data[f'loss_{i}'] = loss_list[i].item()
            logger_data['lr'] = optimizer.param_groups[0]["lr"]
            metric_logger.update(logger_data)
    else:
        for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):
            batch = dynamic_mask(batch, cfg.mask_ratio, cfg.same_column)
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad()
            # loss_ita, loss_mfm, loss_mlm = model(batch)
            loss, loss_list = model(batch)
            assert len(loss_list) == loss_num

            loss.backward()
            optimizer.step()
            if step == "batch":
                lr_scheduler.step()
            for i in range(loss_num):
                logger_data[f'loss_{i}'] = loss_list[i].item()
            logger_data['lr'] = optimizer.param_groups[0]["lr"]
            metric_logger.update(logger_data)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  


if __name__ == '__main__':
    cfg = create_parser()
    # fix seed
    seed = cfg.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    torch.cuda.set_device(cfg.local_rank)
    device = torch.device("cuda", cfg.local_rank)
    dist.init_process_group(backend='nccl')
    
    data_type = cfg.dataset
    model = None
    if not cfg.use_mask_loss:
        loss_num = 1
    elif cfg.use_mfm and cfg.use_mlm:
        loss_num = 3
    else:
        loss_num =2
    
    train_struct,valid_struct,train_text,valid_text, struct_data = make_train_valid_dfs(cfg.struct_path, cfg.text_path, seed, data_type)
    linear_feature_columns, dnn_feature_columns, train_struct_input, valid_struct_input = \
        process_struct_data(data_type,train_struct,valid_struct,struct_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_tokenizer, local_files_only=True)
    num_added_tokens = tokenizer.add_tokens('[val]')
  
    train_loader, train_sampler = build_loaders(train_struct_input, train_text,
                                 linear_feature_columns, dnn_feature_columns,tokenizer, mode='train')
    valid_loader, _ = build_loaders(valid_struct_input, valid_text,
                                 linear_feature_columns,dnn_feature_columns,tokenizer, mode ='valid')
    
    with open(cfg.meta_path) as fh:
        meta_data = json.load(fh)
    total_feature_num = meta_data['feature_num']
    
    if is_main_process():
        print('world size', dist.get_world_size())
        print('train len', len(train_struct), ', valid_len', len(valid_struct))
        print("len_tokenizer:", len(tokenizer))
        print("total feature num: ", total_feature_num)

    model = MaskCTR(cfg, cfg.rec_embedding_dim, cfg.text_embedding_dim, cfg.text_encoder_model,
                struct_linear_feature_columns = linear_feature_columns,
                struct_dnn_feature_columns= dnn_feature_columns,
                struct_feature_num=total_feature_num,
                text_tokenizer_num=len(tokenizer)-1).to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[cfg.local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=1.0, end_factor=.0, total_iters=cfg.epochs)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.epochs, eta_min=0)

    step = "epoch"
    best_loss = float('inf')

    restore_path = cfg.output_prefix_path+ str(data_type)+ f'{cfg.temperature}_{cfg.use_mfm}_{cfg.use_mlm}_{cfg.epochs}_{cfg.lr}_'
    for epoch in range(cfg.epochs):
        # shuffle
        train_sampler.set_epoch(epoch)
        model.train()
        train_stats = train_epoch(model, train_loader, optimizer, lr_scheduler, step)
        lr_scheduler.step()
        
        if is_main_process():
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }                     
            torch.save(model.module, restore_path +"best.pt")
            torch.save(model.module.rec_encoder,restore_path +"rec_best.pt")
            torch.save(model.module.text_encoder, restore_path +"text_best.pt")
            torch.save(model.module.rec_projection, restore_path +"rec_projection.pt")
            torch.save(model.module.text_projection, restore_path +"text_projection.pt")
            with open(restore_path+ "log.txt","a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier() 
