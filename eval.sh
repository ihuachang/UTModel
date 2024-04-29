#!/bin/bash

# Common parameters
batch_size=32
dataset_path="/data2/peter/auto"
model_path="/data2/peter/model/aiw/VL2DModel_heatmap/model_7.pth"
model_name="VL2DModel"
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