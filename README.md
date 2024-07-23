# FLIP: Fine-grained Alignment between ID-based Models and Pretrained Language Models for CTR Prediction
This is the pytorch implementation of ***FLIP*** proposed in the paper [FLIP: Fine-grained Alignment between ID-based Models and Pretrained Language Models for CTR Prediction](https://arxiv.org/abs/2310.19453). (RecSys'24)

## Requirements
~~~python
pip install -r requirments.txt
~~~

## First Step: pretrain FLIP
To pretrain FLIP, please run
`run_script.py`


## Second Step: finetune FLIP
To finetune the ID-based model ($\text{FLIP}_{ID}$), please run `finetune_ctr.py`

To finetune the PLM-based model ($\text{FLIP}_{PLM}$), please run `finetune_nlp.py`

To finetune both the ID-based model and PLM-based model (FLIP), please run `funetine_all.py`

## How to run baselines
We also provide shell scripts for baselines.

To run the `ID-based model` baseline:
~~~python
python ctr_base.py
~~~
To run the `PLM-based model` baseline:
~~~python
python ctr_bert.py
python ptab.py
~~~

