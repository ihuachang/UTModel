#!/bin/bash

# Common parameters
BATCH_SIZE=4
DATASET_PATH="/data/peter/animation/dataset"

CSV_PATH="./exp/pick.csv"  # Uncomment this line if you need to specify a CSV path

# UNET 3D Models
UNET_3D_MODELS=("0.0_0.0/model_4.pth" "0.0_0.5/model_6.pth" "0.0_1.0/model_6.pth" "0.0_1.5/model_5.pth" "0.0_2.0/model_7.pth" "1.0_0.0/model_6.pth" "1.0_1.0/model_8.pth" "1.0_2.0/model_7.pth" "2.0_0.0/model_5.pth" "2.0_1.0/model_8.pth")

# UNET 2D Models
# UNET_2D_MODELS=("4_1/model_10.pth" "4_2/model_9.pth" "4_3/model_9.pth" "4_4/model_10.pth")

# Function to run the eval script
run_eval() {
    local model_path=$1
    local model_type=$2
    python eval.py \
        --batch_size $BATCH_SIZE \
        --model_path $model_path \
        --dataset_path $DATASET_PATH \
        --model_type $model_type \
        # --csv_path $CSV_PATH  # Uncomment this line if you need to specify a CSV path
}

# Evaluate UNET 3D Models
for model in "${UNET_3D_MODELS[@]}"; do
    run_eval "/data/peter/animation/checkpoint_413/UNET3D/$model" "UNET3D"
done

# Evaluate UNET 2D Models
# for model in "${UNET_2D_MODELS[@]}"; do
#     run_eval "/home/ihua/replay/checkpoints/UNET2D/$model" "UNET2D"
# done
