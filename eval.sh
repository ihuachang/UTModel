#!/bin/bash

# Common parameters
batch_size=32
dataset_path="/data2/peter/validation_set/rico"
# dataset_path="/data2/peter/rico"
model_path="/data2/peter/model/rico/VLModel_heatmap/model_17.pth"
model_name="VLModel"
save_path="./validation"
decoder_name="heatmap"
csv_path="./valid.csv"

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