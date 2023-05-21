#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
max_len=128
batch_size=32
max_epochs=100
lr=0.0008
gamma=0.9
dropout=0.5
num_layers=1
train_pattern=gbgb
file_path=/home/yaoshuai/tools/BGC_labels_pred/lstm_bgc/data/BGC_TD_dataset_10.csv
nohup python ./src/train.py --max_len $max_len --batch_size $batch_size --max_epochs $max_epochs --lr $lr --gamma $gamma --dropout $dropout --num_layers $num_layers --train_pattern $train_pattern --file_path $file_path \
    >/home/yaoshuai/tools/lstm_bgc/formal/out_save/${train_pattern}_${max_len}_${batch_size}_${max_epochs}_${lr}v${gamma}_${dropout}_${num_layers}.out 2>&1 &
echo "Done"