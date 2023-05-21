#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pt
 model_path=/home/yaoshuai/tools/lstm_bgc/formal/model_save/gene/GBgb_128_32_0.0008v0.9_0.5_1_f4.pt
 file_path=/home/yaoshuai/data/validation_data/9genomes/formal/sentence_out/9genomes.csv
 max_len=128
 batch_size=512
 out_factor=9genomes_
 model_name=${model_path##*/}
 python -u ./src/test.py --model_path $model_path --file_path $file_path --max_len $max_len --batch_size $batch_size --out_factor $out_factor \
     >./out_save/test/${out_factor}${max_len}_${model_name%.*}.out 2>&1 &