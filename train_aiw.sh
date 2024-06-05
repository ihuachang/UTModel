#!/bin/bash

# Set paths and parameters
dataset_path="/data2/peter/aiw"
save_path="/data2/peter/model"
epochs=100
lr=0.0001
csv_path="./exp/pick.csv"
# model_names=("VLModel" "VL2DModel" "UNet")  # List of model names
# model_names=("UNet2D" "VL2DModel" "LModel")  # List of model names
model_names=("UNet2D" "VL2DModel" "LModel")
# lmodel="/home/ihua/UTModel/test/aiw/LModel_heatmap/LAModel/LModel_LAModel_2.pth"
freeze=0
# Loop over models
for model_name in "${model_names[@]}"
do
    echo "Training model: ${model_name}"

    # Set batch size based on the model
    if [ "$model_name" = "VLModel" ]; then
        batch_size=50
    elif [ "$model_name" = "VL2DModel" ]; then
        batch_size=32
        val_batch_size=64
    elif [ "$model_name" = "UNet" ]; then
        batch_size=40
    elif [ "$model_name" = "LModel" ]; then
        batch_size=128
        val_batch_size=256
    elif [ "$model_name" = "UNet2D" ]; then
        batch_size=64
        val_batch_size=128
    fi
    # Loop over loss_alpha and loss_gamma values
    for loss_alpha in 4  # Changed from seq syntax for simplicity
    do
        for loss_gamma in 4
        do
            echo "Training with alpha: ${loss_alpha}, gamma: ${loss_gamma}"
            python train.py \
                --epochs ${epochs} \
                --batch_size ${batch_size} \
                --lr ${lr} \
                --decoder "point" \
                --dataset_path ${dataset_path} \
                --model_name ${model_name} \
                --loss_alpha ${loss_alpha} \
                --loss_gamma ${loss_gamma} \
                --save_path ${save_path} \
                --val_batch_size ${val_batch_size} \
                --freeze ${freeze} \
                --gpu 1
        done
    done
done