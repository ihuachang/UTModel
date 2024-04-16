#!/bin/bash

# dataset_path="/data/peter/animation/dataset/"
dataset_path="/data2/peter/"
save_path="/home/ihua/VLM/test/checkpoint"
epochs=16
batch_size=50
lr=0.00001
csv_path="./exp/pick.csv"

for loss_alpha in $(seq 2 1 5)
do
    for loss_gamma in $(seq 2 1 5)
    do
        python train.py \
            --epochs ${epochs} \
            --batch_size ${batch_size} \
            --lr ${lr} \
            --dataset_path ${dataset_path} \
            --loss_alpha ${loss_alpha} \
            --loss_gamma ${loss_gamma} \
            --save_path ${save_path} \
            # --csv_path ${csv_path} \
    done
done