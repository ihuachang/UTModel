#!/bin/bash

# Common parameters
batch_size=32
dataset_path="/data2/peter/aiw"
model_path="/data2/peter/model/aiw/UNet2D_heatmap/model_1.pth"
model_name="UNet2D"
save_path="./validation"
decoder_name="heatmap"


python3 eval.py \
    --model_path ${model_path} \
    --batch_size ${batch_size} \
    --save_path ${save_path} \
    --decoder "heatmap" \
    --dataset_path ${dataset_path} \
    --model_name ${model_name} \
    --decoder ${decoder_name} \
    --gpu 1