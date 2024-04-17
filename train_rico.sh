#!/bin/bash

# Set paths and parameters
dataset_path="/data2/peter/rico"
save_path="/data2/peter/model"
epochs=50
lr=0.001
csv_path="./exp/pick.csv"

model_names=("VLModel" "VL2DModel" "UNet") # List of model names

# Loop over models
for model_name in "${model_names[@]}"
do
    echo "Training model: ${model_name}"

    # Set batch size based on the model
    if [ "$model_name" = "VLModel" ]; then
        batch_size=50
    elif [ "$model_name" = "VL2DModel" ]; then
        batch_size=100
    elif [ "$model_name" = "UNet" ]; then
        batch_size=40
    fi
    # Loop over loss_alpha and loss_gamma values
    for loss_alpha in 3 4 5 # Changed from seq syntax for simplicity
    do
        for loss_gamma in 3 4 5
        do
            echo "Training with alpha: ${loss_alpha}, gamma: ${loss_gamma}"
            python train.py \
                --epochs ${epochs} \
                --batch_size ${batch_size} \
                --lr ${lr} \
                --dataset_path ${dataset_path} \
                --model_name ${model_name} \
                --loss_alpha ${loss_alpha} \
                --loss_gamma ${loss_gamma} \
                --save_path ${save_path} \
                --csv_path ${csv_path} \
                --test 1 
        done
    done
done
