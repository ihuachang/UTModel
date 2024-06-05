# Project Name

Description of your project goes here.

## Environment Setup

To set up the environment, use the provided `environment.yml` file. Run the following command:

```bash
conda env create -f environment.yml
```
This will create a Conda environment with all the necessary dependencies for the project.

## Training

Modify the `train_origin.sh` script to fit your task.
Ensure that the supported model names are defined in `train.py`:
```python
models = {
    "ULModel": ULModel,
    "VLModel": VLModel,
    "VL2DModel": VL2DModel,
    "UNet": UNet,
    "UNet3D": UNet3D,
    "UNet2D": UNet2D,
    "LModel": LModel
}
```

## Evaluation
Modify the content in `eval.sh` to evaluate your test dataset.
