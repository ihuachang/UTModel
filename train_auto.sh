#!/bin/bash

# Set paths and parameters
dataset_path="/data2/peter/auto_dataset/62dataset"
test_dataset_path="/data2/peter/validation_set/manual/origin"
save_path="/data2/peter/test_model"
epochs=100
lr=0.0001
csv_path="./exp/pick.csv"
# model_names=("VLModel" "VL2DModel" "UNet")  # List of model names
# model_names=("UNet2D" "VL2DModel" "LModel")  # List of model names
model_names=("ULModel")
lmodel="/data2/peter/model/aiw/VL2DModel_heatmap/LAModel/VL2DModel_LAModel_8.pth"
unet3d="/data2/peter/model/rico/UNet3D_heatmap/UNET3D/UNet3D_BLOCK3D_9.pth"
loss_alpha=4
loss_gamma=4
# Loop over models
for model_name in "${model_names[@]}"
do
    echo "Training model: ${model_name}"

    # Set batch size based on the model
    if [ "$model_name" = "VLModel" ]; then
        batch_size=64
        val_batch_size=64
    elif [ "$model_name" = "VL2DModel" ]; then
        batch_size=32
        val_batch_size=64
    elif [ "$model_name" = "UNet" ]; then
        batch_size=64
        val_batch_size=128
    elif [ "$model_name" = "LModel" ]; then
        batch_size=64
        val_batch_size=128
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
        --gpu 0 \
        --test_dataset_path ${test_dataset_path} \
        # --lamodel_path ${lmodel} \
        # --unet3d_path ${unet3d} \
        # --freeze \

done

