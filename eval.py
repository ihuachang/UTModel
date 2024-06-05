import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.cuda.amp import autocast
from tqdm import tqdm
import argparse
import os
import json
import csv

from tools.hp5dataset import MultiFileDataset as Dataset
from model.models import VLModel, VL2DModel, UNet, UNet3D, UNet2D, LModel
from tools.utils import load_blockla_parameters, validate
from tools.loss import FocalLoss, BBoxLoss

models = {
    "VLModel": VLModel,
    "VL2DModel": VL2DModel,
    "UNet": UNet,
    "UNet3D": UNet3D,
    "UNet2D": UNet2D,
    "LModel": LModel
}

decoders = {
    "heatmap": "heatmap",
    "point": "point"
}

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_name = args.dataset_path.split('/')[-1]
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_path = os.path.join(args.save_path, f"validation_{dataset_name}.csv")

    model = models[args.model_name]()
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)
    print(f"Model {args.model_name} loaded successfully.")

    if args.lamodel_path:
        load_blockla_parameters(model, args.lamodel_path)
        print("LAModel loaded successfully.")
    
    dataset = Dataset(data_dir=args.dataset_path, type="test", csv_file=args.csv_path, decode_type=args.decoder)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.decoder == "point":
        criterion = BBoxLoss()
    else:
        criterion = FocalLoss(args.loss_alpha, args.loss_gamma)

    validation_loss, heat_validation_precision = validate(model, data_loader, criterion, device, args.decoder)
    print(f"Validation Precision: {heat_validation_precision:.4f}, loss: {validation_loss:.4f}")

    # Write results to CSV
    if not os.path.exists(save_path):
        with open(save_path, "w") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(["Model", "Dataset", "Precision"])
    
    with open(save_path, "a") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([args.model_path.split("/")[-3:], dataset_name, heat_validation_precision, validation_loss])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a model")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="auto")
    parser.add_argument("--lamodel_path", type=str)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--csv_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--model_name", type=str, choices=models.keys(), help="Model to use for validation")
    parser.add_argument("--decoder", type=str, choices=decoders.keys(), default="heatmap")
    parser.add_argument("--loss_alpha", type=float, default=4)
    parser.add_argument("--loss_gamma", type=float, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()
    main(args)
