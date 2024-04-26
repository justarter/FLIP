import sys
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',type=str, default='movielens')  # movielens bookcrossing goodreads
parser.add_argument('--backbone', type=str, default='DCNv2') # DeepFM AutoInt DCNv2
parser.add_argument('--llm', type=str, default='tiny-bert') # 'tiny-bert'  'roberta' 'roberta-large'
parser.add_argument('--epochs', type=int, default=30)

add_args = parser.parse_args()

TARGET_PY_FILE = 'pretrain_MaskCTR_ddp.py'

NUM_GPU=8

PORT_ID=15637
PREFIX = f'python -m torch.distributed.launch --nproc_per_node {NUM_GPU} --master_port {PORT_ID} {TARGET_PY_FILE}'.split(" ")

if add_args.llm == 'tiny-bert':
    MIXED_PRECISION=False # no need mixed
    batch_size = 128
elif add_args.llm == 'roberta':
    MIXED_PRECISION=True
    batch_size = 64
elif add_args.llm == 'roberta-large':
    MIXED_PRECISION=True
    batch_size = 16
    
SAMPLE=False
            
for EPOCHS in [add_args.epochs]:
    for BS in [batch_size]:
        for DATASET in [add_args.dataset]:
            for TEM in [0.7]:
                for LR in [1e-4]:
                    for USE_MFM in [True]:
                        for USE_MLM in [True]:
                            for BACKBONE in [add_args.backbone]:
                                for USE_ATTENTION in [True]:
                                    subprocess.run(PREFIX + [
                                        f'--init_method={add_args.init_method}',
                                        f'--train_url={add_args.train_url}',
                                        f'--backbone={BACKBONE}',
                                        f'--temperature={TEM}',
                                        f'--use_mfm={USE_MFM}',
                                        f'--use_mlm={USE_MLM}',
                                        f'--epochs={EPOCHS}',
                                        f'--lr={LR}',
                                        f'--batch_size={BS}',
                                        f'--dataset={DATASET}',
                                        f'--sample={SAMPLE}',
                                        f'--mixed_precision={MIXED_PRECISION}',
                                        f'--llm={add_args.llm}',
                                        f'--use_attention={USE_ATTENTION}'
                                    ])

