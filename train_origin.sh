#!/bin/bash

# Set paths and parameters
dataset_path=""
save_path=""
epochs=100
lr=0.0001
csv_path="./exp/pick.csv"
model_names=("ULModel" "VLModel" "VL2DModel" "UNet") # List of model names

lmodel=""
freeze=0
# Loop over models
for model_name in "${model_names[@]}"
do
    echo "Training model: ${model_name}"

    # Set batch size based on the model
    if [ "$model_name" = "VLModel" ]; then
        batch_size=16
        val_batch_size=32
    elif [ "$model_name" = "VL2DModel" ]; then
        batch_size=64
        val_batch_size=128
    elif [ "$model_name" = "UNet" ]; then
        batch_size=16
        val_batch_size=32
    elif [ "$model_name" = "LModel" ]; then
        batch_size=128
        val_batch_size=256
    elif [ "$model_name" = "UNet2D" ]; then
        batch_size=64
        val_batch_size=128
    elif [ "$model_name" = "UNet3D" ]; then
        batch_size=16
        val_batch_size=32
    elif [ "$model_name" = "ULModel" ]; then
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
                --decoder "heatmap" \
                --dataset_path ${dataset_path} \
                --model_name ${model_name} \
                --loss_alpha ${loss_alpha} \
                --loss_gamma ${loss_gamma} \
                --save_path ${save_path} \
                --val_batch_size ${val_batch_size} \
                --gpu 1 \
                # --freeze ${freeze} \
    
        done
    done
done