import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable

from tqdm import tqdm
import argparse
import os, csv

from tools.loss import FocalLoss
from tools.hp5dataset import MultiFileDataset as Dataset
from tools.hp5dataset import custom_collate_fn
from tools.model import VLModel, VL2DModle, UNet

# Model dictionary to dynamically select the model
models = {
    "VLModel": VLModel,
    "VL2DModel": VL2DModle,
    "UNet": UNet
}

def check_clicks(outputs, labels):
    # Get the indices of the max points in the outputs
    max_indices = torch.argmax(outputs.view(outputs.shape[0], -1), dim=1)

    # Compute the coordinates from the indices
    max_points = torch.stack((max_indices // outputs.shape[3], max_indices % outputs.shape[3]), dim=1)

    # Get the corresponding values from the labels
    corresponding_label_values = labels[torch.arange(outputs.shape[0]), 0, max_points[:, 0], max_points[:, 1]]

    # Count the number of t?imes the corresponding label value is greater than 0.95
    correct = torch.sum(corresponding_label_values > 0.0001).item()

    # Compute the precision
    precision = correct / outputs.shape[0]

    return precision

def evaluate(args):
    # Configuration
    batch_size = args.batch_size
    model_path = args.model_path
    csv_path = args.csv_path
    model_type = args.model_type
    dataset_path = args.dataset_path

    # Prepare your data loader
    dataset = Dataset(data_dir=dataset_path, train=False, csv=csv_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn = custom_collate_fn)

    # Load your model
    if args.model_name not in models:
        raise ValueError("Model not supported")
    model = models[args.model_name]()

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = FocalLoss()

    valid_name = f"{model_type}_{model_path.split('/')[-2]}_{model_path.split('/')[-1].split('.')[0]}"
    csv_name = f"valid_{csv_path.split('/')[-1].split('.')[0]}"

    if not os.path.exists("valid"):
        os.makedirs("valid")

    if not os.path.exists(f"valid/{csv_name}.csv"):
        with open(f'valid/{csv_name}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Model_name", "Validation Loss", "Precision"])

    model.load_state_dict(torch.load(model_path))

    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_precision = 0
        progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
        for i, (text, bound, mask, input2, labels) in progress_bar:
            text, bound, mask, input2, labels = Variable(text).to(device), Variable(bound).to(device), Variable(mask).to(device), Variable(input2).to(device), Variable(labels).to(device)

            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            precision = check_clicks(outputs, labels)
            total_loss += loss.item()
            total_precision += precision
        
        with open(f'valid/{csv_name}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([valid_name, total_loss/len(data_loader), total_precision/len(data_loader)])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a UNet model")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--model_path", type=str, default=".", help="path to the trained model")
    parser.add_argument("--csv_path", type=str, default=None, help="path to the csv file")
    parser.add_argument("--model_type", type=str, default="UNET3D", help="2d or 3d model")
    parser.add_argument("--dataset_path", type=str, default=".", help="path to the dataset")

    args = parser.parse_args()
    evaluate(args)
