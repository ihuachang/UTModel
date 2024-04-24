import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.autograd import Variable
from torch.cuda.amp import GradScaler, autocast
from numpy import nan
from tqdm import tqdm
import argparse
import os 
import json
import shutil

from tools.loss import FocalLoss, BBoxLoss
from tools.hp5dataset import MultiFileDataset as Dataset
from tools.hp5dataset import heatmap_collate_fn, point_collate_fn
from model.models import VLModel, VL2DModel, UNet, UNet3D, UNet2D, LModel
from tools.valid import check_clicks
from tools.utils import EarlyStopper

# Model dictionary to dynamically select the model
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

Random = 880323
H5SIZE = 16384

# Assuming you are loading a pre-trained LAModel
def load_blockla_parameters(model, load_path, freeze=True):
    model.laModel.load_state_dict(torch.load(load_path))
    if freeze:
        for param in model.laModel.parameters():
            param.requires_grad = False

def validate(model, data_loader, criterion, device, decoder_name):
    model.eval()
    validation_loss = 0.0
    total_precision = 0.0

    with torch.no_grad(), autocast():
        for text, bound, mask, input2, labels, heats in tqdm(data_loader, total=len(data_loader), desc="Validating"):
            text, bound, mask, input2, labels, heats = (Variable(tensor).to(device) for tensor in [text, bound, mask, input2, labels, heats])

            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            if decoder_name == "point":
                labels = heats
            precision = check_clicks(outputs, labels, decoder_name)

            validation_loss += loss.item()
            total_precision += precision

    validation_loss /= len(data_loader)
    validation_precision = total_precision / len(data_loader)
    return validation_loss, validation_precision

def train(args):
    # Configuration
    num_epochs = args.epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    learning_rate = args.lr
    dataset_path = args.dataset_path
    model_path = args.model_path
    lamodel_path = args.lamodel_path
    csv_path = args.csv_path
    loss_alpha = args.loss_alpha
    loss_gamma = args.loss_gamma
    save_path = args.save_path
    decoder_name = args.decoder
    test = args.test
    gpu = args.gpu

    save_path = f"{save_path}/{dataset_path.split('/')[-1]}/{args.model_name}_{decoder_name}/{loss_alpha}_{loss_gamma}"
    if test:
        save_path = f"./test/{dataset_path.split('/')[-1]}/{args.model_name}_{decoder_name}/{loss_alpha}_{loss_gamma}"
    if args.model_name not in models:
        raise ValueError("Model not supported")
    
    model = models[args.model_name](decoder_name)

    if decoder_name == "heatmap":
        print("Using heatmap decoder")
        custom_collate_fn = heatmap_collate_fn
    else:
        print("Using point decoder")
        custom_collate_fn = point_collate_fn

    # choose gpu 0 or 1
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

    # fix random seeds for reproducibility
    torch.manual_seed(Random)
    torch.cuda.manual_seed(Random)

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    if not os.path.exists(os.path.join(save_path, "LAModel")):
        os.makedirs(os.path.join(save_path, "LAModel"))

    config = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Prepare your data loader
    train_dataset = Dataset(data_dir=dataset_path, type="train", csv_file=csv_path, demo=test, decode_type=decoder_name)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    valid_dataset = Dataset(data_dir=dataset_path, type="val", csv_file=csv_path, demo=test, decode_type=decoder_name)
    valid_data_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    test_dataset = Dataset(data_dir=dataset_path, type="test", csv_file=csv_path, demo=test, decode_type=decoder_name)
    test_data_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    
    if lamodel_path is not None:
        load_blockla_parameters(model, lamodel_path)
        print("LAModel loaded successfully")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # Define loss function and optimizer
    if decoder_name == "heatmap":
        criterion = FocalLoss(loss_alpha, loss_gamma)
    else:
        criterion = BBoxLoss()

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,  # Learning rate
        weight_decay=0.01  # Weight decay
    )
    scaler = GradScaler()

    import csv

    with open(f'{save_path}/losses,{loss_alpha},{loss_gamma}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # write the header
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Valid Precision", "Test Precision"])

    torch.save(model.state_dict(), f'{save_path}/model_{0}.pth')

    early_stopper = EarlyStopper(patience=4, min_delta=0.00001)

    for epoch in range(num_epochs):
        train_dataset.shuffle_blocks(block_size=H5SIZE)
        # print the first item in the dataset
        # print(train_dataset.dataset_keys[0])
        # Instantiate a progress bar object with the total length equal to the size of the data loader
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        
        training_loss = 0.0

        for i, (text, bound, mask, input2, labels, heats) in progress_bar:
            with autocast():  # Enable automatic mixed precision
                if args.model_name == "LModel":
                    text, bound, mask, labels = text.to(device), bound.to(device), mask.to(device), labels.to(device)
                    outputs = model(text, bound, mask)
                else:
                    text, bound, mask, input2, labels = text.to(device), bound.to(device), mask.to(device), input2.to(device), labels.to(device)
                    outputs = model(text, bound, mask, input2)

                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Scale the loss and call backward
            scaler.step(optimizer)  # Unscales gradients and step optimizer
            scaler.update()  # Update the scaler for the next iteration
            training_loss += loss.item()
            progress_bar.set_description(f'Epoch: {epoch+1}/{num_epochs}')
   
        training_loss = training_loss / len(train_data_loader)  # get average loss
        progress_bar.close()
        torch.save(model.state_dict(), f'{save_path}/model_{epoch+1}.pth')
        # Checkpoint saving example for extracting sub-models
        if hasattr(model, 'laModel'):
            torch.save(model.laModel.state_dict(), os.path.join(save_path, "LAModel", f'LAModel_checkpoint{epoch+1}.pth'))

        # Validate the model
        validation_loss, precision = validate(model, valid_data_loader, criterion, device, decoder_name)
        _, test_precision = validate(model, test_data_loader, criterion, device, decoder_name)

        print(f'Epoch: {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Valid Precision: {precision:.4f}, Test Precision: {test_precision:.4f}')
        model.train()  # set the model back to training mode

        # Write the losses to the csv file
        with open(f'{save_path}/losses,{loss_alpha},{loss_gamma}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, training_loss, validation_loss, precision, test_precision])
        
        if early_stopper.early_stop(validation_loss):             
            break
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--val_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--model_path", type=str, help="Path to a pre-trained model", default=None)
    parser.add_argument("--model_name", type=str, choices=models.keys(), help="Model to use for training")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file", default=None)
    parser.add_argument("--loss_alpha", type=int, default=2)
    parser.add_argument("--loss_gamma", type=int, default=2)
    parser.add_argument("--save_path", type=str, default=".")
    parser.add_argument("--test", type=bool, default=False)
    parser.add_argument("--decoder", type=str, choices=decoders.keys(), default="heatmap")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--lamodel_path", type=str, help="Path to a pre-trained LAModel", default=None)
    args = parser.parse_args()
    train(args)
