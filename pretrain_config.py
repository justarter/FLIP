import argparse
from utils import str2bool

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type = int, default=2012)

    parser.add_argument("--sample", type = str2bool, default=False, help = "sample dataset") 
    # mixed precision
    parser.add_argument("--mixed_precision", type = str2bool, default=False) 

    parser.add_argument('--dataset', default='movielens', type=str, help='dataset name')
    parser.add_argument('--backbone', default='DeepFM', type=str, help='')
    parser.add_argument("--llm", type = str, default='tiny-bert') 
    # temperature
    parser.add_argument("--temperature", type = float, default=0.7)
    parser.add_argument("--batch_size", type = int, default=128) 
    parser.add_argument("--epochs", type = int, default=25) 

    parser.add_argument("--lr", type = float, default=5e-5)
    parser.add_argument("--weight_decay", type = float, default=1e-3) 
    parser.add_argument("--num_workers", type = int, default=4)
    # ablation
    parser.add_argument("--use_contrastive", type = str2bool, default=True)
    parser.add_argument("--use_mfm", type = str2bool, default=True)
    parser.add_argument("--use_mlm", type = str2bool, default=True)
    parser.add_argument("--use_attention", type = str2bool, default=False)
    # ddp
    parser.add_argument("--local_rank", type = int, default=0)

    parser.add_argument("--text_mask_ratio", type = float, default=0.15) 
    parser.add_argument("--ctr_mask_ratio", type = float, default=0.15) 
    parser.add_argument("--mask_same_column", type = str2bool, default=False) 
    # nce loss
    parser.add_argument("--pt_neg_num", type = int, default=25)
    parser.add_argument("--pt_loss", type = str, default='nce')
  
    args = parser.parse_args()
    
    args.load_prefix_path = "./"
    args.output_prefix_path = './'

    if args.dataset == 'movielens':
        args.rec_embedding_dim = 384
        args.data_path = args.load_prefix_path + "data/ml-1m/"
        args.max_length = 100

    elif args.dataset == 'bookcrossing':
        args.rec_embedding_dim = 384
        args.data_path = args.load_prefix_path + "data/BookCrossing/"
        args.max_length = 100

    elif args.dataset == 'goodreads': 
        args.rec_embedding_dim = 608
        args.data_path = args.load_prefix_path + "data/GoodReads/"
        args.max_length = 180

    args.text_path =args.data_path + "text.txt"
    args.struct_path = args.data_path + "remap_data.csv"
    args.feat_count_path = args.data_path + 'feat_count.pt'
    args.meta_path = args.data_path + 'meta.json'

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
    # use mask
    args.use_mask_loss = args.use_mfm or args.use_mlm
    for k,v in vars(args).items():
        print(k,'=',v)
    return args
