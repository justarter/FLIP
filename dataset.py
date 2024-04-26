import os
import torch
import numpy as np
import pandas as pd
import random

from preprocessing.inputs import SparseFeat,  build_input_features
from preprocessing.inputs import create_embedding_matrix,   get_feature_names 
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class CtrDataset2(torch.utils.data.Dataset):
    def __init__(self, struct_data, text_data, text_label, linear_feature_columns, dnn_feature_columns, tokenizer, max_length):
        self.max_length = max_length
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        self.rec_data = struct_data
        self.text_label = text_label
        if isinstance(self.rec_data, dict):
            self.rec_data = [self.rec_data[feature] for feature in self.feature_index]
        for i in range(len(self.rec_data)):
            # print(self.rec_data[i].shape) # (B,)
            if len(self.rec_data[i].shape) == 1:
                self.rec_data[i] = np.expand_dims(self.rec_data[i], axis=1)
        self.rec_data = torch.from_numpy(np.concatenate(self.rec_data, axis=-1)).type(torch.float32) # B numfield
        
        self.text_data = list(text_data)
        # print(self.text_data[0]) 
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # return tensor，atten mask
        encoding = self.tokenizer(
            self.text_data[idx],
            add_special_tokens=True,
            truncation = True,
            max_length=self.max_length,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        item['rec_data'] = self.rec_data[idx]
        item['text_data'] = self.text_data[idx]
        item['label'] = torch.tensor(self.text_label[idx], dtype=torch.int)
    
        return item

    def __len__(self):
        return len(self.text_data)
    
    
class CtrDataset3(torch.utils.data.Dataset):
    def __init__(self, struct_data, text_data, text_label, linear_feature_columns, dnn_feature_columns, tokenizer, max_length, mask_ratio):
        self.max_length = max_length
        self.feature_index = build_input_features(linear_feature_columns + dnn_feature_columns)
        self.rec_data = struct_data
        self.text_label = text_label
        if isinstance(self.rec_data, dict):
            self.rec_data = [self.rec_data[feature] for feature in self.feature_index]
        for i in range(len(self.rec_data)):
            if len(self.rec_data[i].shape) == 1:
                self.rec_data[i] = np.expand_dims(self.rec_data[i], axis=1)
        self.rec_data = torch.from_numpy(np.concatenate(self.rec_data, axis=-1)).type(torch.float32) # B numfield
        
        self.text_data = list(text_data)
        self.mask_ratio = mask_ratio
        self.num_field = self.rec_data.shape[1]
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        added_tokens = self.tokenizer.tokenize(self.text_data[idx])
        tokens, masked_tokens = [], []
        is_between_hyphens, mask_flag = False, False
        tmp_index = -1
        mask_num = int(self.num_field * self.mask_ratio)
        mask_index = torch.rand((1, self.num_field))
        mask_index = torch.argsort(mask_index, dim=-1)[:, :mask_num] # 1 masknum
        for token in added_tokens:
            if len(masked_tokens) > self.max_length:
                break
            if token == '[val]':
                is_between_hyphens = not is_between_hyphens
                if is_between_hyphens:
                    tmp_index += 1
                    mask_flag = True if tmp_index in mask_index else False
                else:
                    mask_flag = False
            elif mask_flag:
                tokens.append(token)
                masked_tokens.append(self.tokenizer.mask_token)
            else:
                tokens.append(token)
                masked_tokens.append(token)
        # print('added tokens', added_tokens, 'real tokens', tokens, "masked tokens",mask_index,masked_tokens)

        tokens = tokens[:self.max_length - 2] 
        tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]  
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
        attention_mask = [1] * len(input_ids)  # attention mask
        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length  # pad
        attention_mask += [0] * padding_length  
        # print(input_ids, attention_mask)

        masked_tokens = masked_tokens[:self.max_length - 2] 
        masked_tokens = [self.tokenizer.cls_token] + masked_tokens + [self.tokenizer.sep_token] 
        masked_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens) 
        masked_input_ids += [0] * padding_length 

        input_ids = np.array(input_ids)
        masked_input_ids = np.array(masked_input_ids)
        mask_text_label = np.where(input_ids != masked_input_ids, input_ids, -100)

        item = {
            'input_ids': torch.tensor(input_ids, dtype=torch.int).flatten(),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.int).flatten(),
            'mask_input_ids':torch.tensor(masked_input_ids, dtype=torch.int).flatten(),
            'mask_text_label': torch.tensor(mask_text_label, dtype=torch.int).flatten(),
            'mask_text_index': mask_index.flatten(),
            'rec_data': self.rec_data[idx],
            'label': torch.tensor(self.text_label[idx], dtype=torch.int)
        }

        return item

    def __len__(self):
        return len(self.text_data)


