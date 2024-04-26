import sys
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--init_method', default='')
parser.add_argument('--train_url', default='')

add_args = parser.parse_args()

if len(add_args.train_url):
    TARGET_PY_FILE = '/home/ma-user/modelarts/user-job-dir/MaskCTR_obs/mlm.py'
else:
    TARGET_PY_FILE = 'mlm.py'

#
NUM_GPU=1

PORT_ID=15637
PREFIX = f'python -m torch.distributed.launch --nproc_per_node {NUM_GPU} --master_port {PORT_ID} {TARGET_PY_FILE}'.split(" ")

MIXED_PRECISION=False
SAMPLE=False
            
for EPOCHS in [10]: # GD 5
    for BS in [512]:
        for DATASET in ['toys']:
                for LR in [5e-5]:
                    for AS in [2]:
                            subprocess.run(PREFIX + [
                                f'--epochs={EPOCHS}',
                                f'--lr={LR}',
                                f'--batch_size={BS}',
                                f'--dataset={DATASET}',
                                f'--accumulation_steps={AS}',
                                f'--sample={SAMPLE}',
                                f'--mixed_precision={MIXED_PRECISION}' 
                            ])


