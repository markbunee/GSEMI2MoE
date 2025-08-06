# GSEMI2MoE

## fold

code dir –> GSEMI2MoE



datasets dir –> datasets

|—images

|—labels

​	|—class_task1

​	|—class_task2

​	|—……



models save dir -> models



readme.md



requirements.txt



## train
deepspeed --num_gpus=5 GSEMI2MoE\train.py --deepspeed --deepspeed_config config.json

## envirement

cd GSEMI2MoE\parallel_linear

pip install .

PyTorch  2.1.0

Python  3.10(ubuntu22.04)

CUDA  12.1

pip install -r requirements.txt

