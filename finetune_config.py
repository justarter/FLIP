import torch
import argparse
from utils import str2bool

def create_finetune_nlp_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument("--llm", type = str, default='tiny-bert', help = '' ) 
    parser.add_argument("--sample", type = str2bool, default=False, help = "sample dataset") 
    # mixed precision
    parser.add_argument("--mixed_precision", type = str2bool, default=False) 
    # parameter for pretrain
    parser.add_argument("--model_path", type = str, help = "") 
    parser.add_argument("--temperature", type = str, help = "") # 0.07, 0.1
    parser.add_argument("--use_mfm", type = str2bool, help = "") 
    parser.add_argument("--use_mlm", type = str2bool, help = "") 
    parser.add_argument("--pre_epochs", type = str, help = "") 
    parser.add_argument("--pre_lr", type = str, help = "") 
    # parameter for finetune
    parser.add_argument("--batch_size", type = int, default=128, help = "") 
    parser.add_argument("--epochs", type = int, default=25, help = "")
    parser.add_argument("--lr", type = float, default=5e-5, help = "") 
    parser.add_argument("--num_workers", type = int, default=4, help = "")

    parser.add_argument("--weight_decay", type = float, default=1e-3, help = "") 
    parser.add_argument("--patience", type = int, default=3, help = "") 
    parser.add_argument("--factor", type = float, default=0.95, help = "") 

    parser.add_argument("--dropout", type = float, default=0.2 , help = "") 

    parser.add_argument("--use_special_token", type = str2bool, default=False)

    parser.add_argument("--obs", type = str2bool, default=True, help = "")  
    args = parser.parse_args()

    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    # dp
    args.batch_size = args.batch_size*torch.cuda.device_count()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'movielens':
        args.data_path = args.load_prefix_path+"data/ml-1m/"
        args.max_length = 100
    elif args.dataset == 'bookcrossing':
        args.data_path = args.load_prefix_path+"data/BookCrossing/"
        args.max_length = 100
    elif args.dataset == 'goodreads':
        args.data_path = args.load_prefix_path+"data/GoodReads/"
        args.max_length = 180
        
    args.text_path = args.data_path+"text.txt"
    args.struct_path = args.data_path+"remap_data.csv"

    if args.llm == 'tiny-bert':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_embedding_dim = 312
    elif args.llm == 'roberta':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/roberta-base/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/roberta-base/"
        args.text_embedding_dim = 768
    elif args.llm == 'roberta-large':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/roberta-large/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/roberta-large/"
        args.text_embedding_dim = 1024

    args.sample_ration = 0.01
    for k,v in vars(args).items():
        print(k,'=',v)
    return args

def create_finetune_all_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument("--llm", type = str, default='tiny-bert', help = '' ) 
    parser.add_argument("--backbone", type = str, default='DeepFM', help = "") 
    parser.add_argument("--sample", type = str2bool, default=False, help = "sample dataset") 
    
    parser.add_argument("--mixed_precision", type = str2bool, default=False) 
    
    parser.add_argument("--model_path", type = str, help = "") 
    parser.add_argument("--temperature", type = str, help = "") # 0.07, 0.1
    parser.add_argument("--use_mfm", type = str2bool, help = "") 
    parser.add_argument("--use_mlm", type = str2bool, help = "") 
    parser.add_argument("--pre_epochs", type = str, help = "") 
    parser.add_argument("--pre_lr", type = str, help = "") 

    parser.add_argument("--batch_size", type = int, default=128, help = "") 
    parser.add_argument("--epochs", type = int, default=25, help = "")
    parser.add_argument("--lr", type = float, default=5e-5, help = "") 
    parser.add_argument("--num_workers", type = int, default=4, help = "")

    parser.add_argument("--weight_decay", type = float, default=1e-3, help = "") 
    parser.add_argument("--patience", type = int, default=3, help = "") 
    parser.add_argument("--factor", type = float, default=0.95, help = "") 
    
    parser.add_argument("--use_mlp", type = str2bool, default=False)
    parser.add_argument("--use_cls", type = str2bool, default=False)

    parser.add_argument("--dropout", type = float, default=0.2 , help = "") 

    parser.add_argument("--obs", type = str2bool, default=True, help = "")  
    parser.add_argument("--alpha", type = float, default=0.2 , help = "") 
    parser.add_argument("--real_mask_index", type = int) 
    args = parser.parse_args()

    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    # dp
    args.batch_size = args.batch_size*torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'movielens':
        args.text_path = args.load_prefix_path+"data/ml-1m/text.txt"
        args.struct_path = args.load_prefix_path+"data/ml-1m/remap_data.csv"
        args.meta_path = args.load_prefix_path+"data/ml-1m/meta.json"
        args.max_length = 100
    elif args.dataset == 'bookcrossing':
        args.text_path = args.load_prefix_path+"data/BookCrossing/text.txt"
        args.struct_path = args.load_prefix_path+"data/BookCrossing/remap_data.csv"
        args.meta_path = args.load_prefix_path+"data/BookCrossing/meta.json"
        args.max_length = 100
    elif args.dataset == 'goodreads':
        args.text_path = args.load_prefix_path+"data/GoodReads/text.txt"
        args.struct_path = args.load_prefix_path+"data/GoodReads/remap_data.csv"
        args.max_length = 180
    
    if args.llm == 'tiny-bert':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_embedding_dim = 312
    elif args.llm == 'roberta':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/roberta-base/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/roberta-base/"
        args.text_embedding_dim = 768
    elif args.llm == 'roberta-large':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/roberta-large/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/roberta-large/"
        args.text_embedding_dim = 1024

    args.sample_ration = 0.01
    for k,v in vars(args).items():
        print(k,'=',v)
    return args





