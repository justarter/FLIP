import torch
import argparse
from utils import str2bool

def create_mlm_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default=2012)

    parser.add_argument("--mixed_precision", type = str2bool, default=False) 
 
    parser.add_argument("--local_rank", type = int, default=0)

    parser.add_argument("--accumulation_steps", type = int, default=2)
    
    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument("--llm", type = str, default='tiny-bert', help = '' ) 
    parser.add_argument("--sample", type = str2bool, default=False, help = "sample dataset") 

    parser.add_argument("--batch_size", type = int, default=128, help = "") 
    parser.add_argument("--epochs", type = int, default=10, help = "")
    parser.add_argument("--lr", type = float, default=5e-5 , help = "")

    parser.add_argument("--weight_decay", type = float, default=1e-3, help = "") 
    parser.add_argument("--num_workers", type = int, default=4, help = "")

    args = parser.parse_args()
    
    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'movielens':
        args.data_path = args.load_prefix_path + 'data/ml-1m/'
        args.max_length = 100
    elif args.dataset == 'bookcrossing':
        args.data_path = args.load_prefix_path + 'data/BookCrossing/'
        args.max_length = 100
    elif args.dataset == 'goodreads':
        args.data_path = args.load_prefix_path + 'data/GoodReads/'
        args.max_length = 180

    args.text_path = args.data_path + "text.txt"
    args.struct_path = args.data_path + "remap_data.csv"
    
    if args.llm == 'tiny-bert':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_embedding_dim = 312
        
    args.sample_ration = 0.01
    for k,v in vars(args).items():
        print(k,'=',v)
    return args


def create_ptab_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mixed_precision", type = str2bool, default=False) 
    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument("--llm", type = str, default='tiny-bert', help = '' ) 
    parser.add_argument("--sample", type = str2bool, default=False, help = "sample dataset") 

    parser.add_argument("--batch_size", type = int, default=128, help = "") 
    parser.add_argument("--epochs", type = int, default=10, help = "")
    parser.add_argument("--lr", type = float, default=5e-5 , help = "") 

    parser.add_argument("--weight_decay", type = float, default=1e-3, help = "") 
    parser.add_argument("--num_workers", type = int, default=4, help = "")

    parser.add_argument("--dropout", type = float, default=0.2 , help = "") 

    args = parser.parse_args()
    
    args.load_prefix_path = "./"
    args.output_prefix_path = './'
 
    args.batch_size = args.batch_size*torch.cuda.device_count()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'movielens':
        args.data_path = args.load_prefix_path + 'data/ml-1m/'
        args.max_length = 100
    elif args.dataset == 'bookcrossing':
        args.data_path = args.load_prefix_path + 'data/BookCrossing/'
        args.max_length = 100
    elif args.dataset == 'goodreads':
        args.data_path = args.load_prefix_path + 'data/GoodReads/'
        args.max_length = 180

    args.text_path = args.data_path + "text.txt"
    args.struct_path = args.data_path + "remap_data.csv"

    if args.llm == 'tiny-bert':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_embedding_dim = 312
        
    args.sample_ration = 0.01
    for k,v in vars(args).items():
        print(k,'=',v)
    return args

def create_ctrbert_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mixed_precision", type = str2bool, default=False) 
    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument("--llm", type = str, default='tiny-bert', help = '' ) 
    parser.add_argument("--sample", type = str2bool, default=False, help = "sample dataset") 
    
    parser.add_argument("--batch_size", type = int, default=128, help = "") 
    parser.add_argument("--epochs", type = int, default=10, help = "")
    parser.add_argument("--lr", type = float, default=5e-5 , help = "") 

    parser.add_argument("--weight_decay", type = float, default=1e-3, help = "") 
    parser.add_argument("--num_workers", type = int, default=4, help = "")

    parser.add_argument("--dropout", type = float, default=0.2 , help = "") # for pred layer
    args = parser.parse_args()
    
    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    # dp
    args.batch_size = args.batch_size*torch.cuda.device_count()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'movielens':
        args.data_path = args.load_prefix_path + 'data/ml-1m/'
        args.max_length = 100
    elif args.dataset == 'bookcrossing':
        args.data_path = args.load_prefix_path + 'data/BookCrossing/'
        args.max_length = 100
    elif args.dataset == 'goodreads':
        args.data_path = args.load_prefix_path + 'data/GoodReads/'
        args.max_length = 180
        
    args.user_text_path = args.data_path+"user_text.txt"
    args.item_text_path = args.data_path+"item_text.txt"
    args.struct_path = args.data_path + "remap_data.csv"

    if args.llm == 'tiny-bert':
        args.text_encoder_model = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_tokenizer = args.load_prefix_path+"pretrained_models/tiny-bert-4l-en/"
        args.text_embedding_dim = 312
        
    args.sample_ration = 0.01
    for k,v in vars(args).items():
        print(k,'=',v)
    return args
