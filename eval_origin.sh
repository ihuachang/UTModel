#!/bin/bash

# Common parameters
batch_size=32
dataset_path=""
model_path=""
model_name="VLModel"
save_path=""
decoder_name="heatmap"
csv_path=""

python3 eval.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --save_path ${save_path} \
    --decoder "heatmap" \
    --dataset_path ${dataset_path} \
    --model_name ${model_name} \
    --decoder ${decoder_name} \
    --gpu 0 \
    # --csv_path ${csv_path}