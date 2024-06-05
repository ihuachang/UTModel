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
from model.models import VLModel, VL2DModel, UNet, UNet3D, UNet2D, LModel, ULModel
from tools.utils import EarlyStopper
from tools.utils import load_blockla_parameters, load_block3d_parameters, load_block2d_parameters
from tools.utils import validate

# Model dictionary to dynamically select the model
models = {
    "ULModel": ULModel,
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

def train(args):
    # Configuration
    num_epochs = args.epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    learning_rate = args.lr
    dataset_path = args.dataset_path
    model_path = args.model_path
    lamodel_path = args.lamodel_path
    unet3d_path = args.unet3d_path
    unet2d_path = args.unet2d_path
    csv_path = args.csv_path
    loss_alpha = args.loss_alpha
    loss_gamma = args.loss_gamma
    save_path = args.save_path
    decoder_name = args.decoder
    test = args.test
    gpu = args.gpu
    freeze = args.freeze
    test_dataset_path = args.test_dataset_path

    print(freeze)

    save_folder = os.path.join(save_path, f"{dataset_path.split('/')[-1]}")
    if test:
        save_folder = os.path.join("./test", f"{dataset_path.split('/')[-1]}")

    model_name = f"{args.model_name}_{decoder_name}"
    if lamodel_path is not None:
        model_name += f"{lamodel_path.split('/')[-1].split('.')[0]}"
    if unet3d_path is not None:
        model_name += f"{unet3d_path.split('/')[-1].split('.')[0]}"
    if unet2d_path is not None:
        model_name += f"{unet2d_path.split('/')[-1].split('.')[0]}"
    if freeze:
        model_name += "_freeze"
    
    save_path = os.path.join(save_folder, model_name)
    
    if args.model_name not in models:
        raise ValueError("Model not supported")
    if lamodel_path is not None:
        save_path = f"{save_path}/{lamodel_path.split('/')[-1]}"
    
    model = models[args.model_name](decoder_name)

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
    
    if not os.path.exists(os.path.join(save_path, "block3d")):
        os.makedirs(os.path.join(save_path, "block3d"))
    
    if not os.path.exists(os.path.join(save_path, "block2d")):
        os.makedirs(os.path.join(save_path, "block2d"))

    config = vars(args)
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Prepare your data loader
    train_dataset = Dataset(data_dir=dataset_path, type="train", csv_file=csv_path, demo=test, decode_type=decoder_name)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    valid_dataset = Dataset(data_dir=dataset_path, type="val", csv_file=csv_path, demo=test, decode_type=decoder_name)
    valid_data_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    print(f"Training on {len(train_dataset)} samples, and Validating on {len(valid_dataset)} samples")

    if test_dataset_path is not None:
        test_dataset = Dataset(data_dir=test_dataset_path, type="test", csv_file=csv_path, demo=test, decode_type=decoder_name)
        test_data_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
    
    if lamodel_path is not None:
        load_blockla_parameters(model, lamodel_path, freeze=freeze)
        print("LAModel loaded successfully")
    
    if unet3d_path is not None:
        load_block3d_parameters(model, unet3d_path, freeze=freeze)
        print("UNet3D loaded successfully")
    
    if unet2d_path is not None:
        load_block2d_parameters(model, unet2d_path, freeze=freeze)
        print("UNet2D loaded successfully")

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

    early_stopper = EarlyStopper(patience=8, min_delta=0.00001)

    for epoch in range(num_epochs):
        train_dataset.shuffle_blocks(block_size=H5SIZE)
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        
        training_loss = 0.0

        for i, (text, bound, mask, input2, bbox, heats) in progress_bar:
            with autocast():  # Enable automatic mixed precision
                if args.model_name == "LModel":
                    text, bound, mask, labels = text.to(device), bound.to(device), mask.to(device), heats.to(device)
                    outputs = model(text, bound, mask)
                else:
                    text, bound, mask, input2, labels = text.to(device), bound.to(device), mask.to(device), input2.to(device), heats.to(device)
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
            torch.save(model.laModel.state_dict(), os.path.join(save_path, "LAModel", f'{args.model_name}_LAModel_{epoch+1}.pth'))

        if hasattr(model, 'block3d'):
            torch.save(model.block3d.state_dict(), os.path.join(save_path, "block3d", f'{args.model_name}_BLOCK3D_{epoch+1}.pth'))
        
        if hasattr(model, 'block2d'):
            torch.save(model.block2d.state_dict(), os.path.join(save_path, "block2d", f'{args.model_name}_BLOCK2D_{epoch+1}.pth'))

        # Validate the model
        validation_loss, precision = validate(model, valid_data_loader, criterion, device, decoder_name)

        test_precision = nan
        if test_dataset_path is not None:
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
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--lamodel_path", type=str, help="Path to a pre-trained LAModel", default=None)
    parser.add_argument("--test_dataset_path", type=str, default=None)
    parser.add_argument("--unet3d_path", type=str, help="Path to a pre-trained UNet3D", default=None)
    parser.add_argument("--unet2d_path", type=str, help="Path to a pre-trained UNet2D", default=None)
    args = parser.parse_args()
    train(args)
