import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable

from tqdm import tqdm
import argparse
import os 
import json

from numpy import nan
import math
from tools.loss import FocalLoss
from tools.hp5dataset import MultiFileDataset as Dataset
from tools.hp5dataset import heatmap_collate_fn, point_collate_fn
from model.models import VLModel, VL2DModel, UNet, UNet3D, UNet2D, LModel
from tools.valid import check_clicks
from tools.utils import EarlyStopper
import shutil

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

# Assuming you are loading a pre-trained LAModel
def load_pretrained_lamodel(model, pretrained_path):
    # Load the state dict for LAModel
    lamodel_state_dict = torch.load(pretrained_path)

    # You should update the keys in lamodel_state_dict if necessary, e.g.:
    # lamodel_state_dict = {'laModel.' + k: v for k, v in lamodel_state_dict.items()}

    # Load the state dict into the model's LAModel component
    model.laModel.load_state_dict(lamodel_state_dict, strict=False)

def validate(model, data_loader, criterion, device, decoder_name):
    progress_bar = tqdm(enumerate(data_loader), total=len(data_loader))
    model.eval()  # set the model to evaluation mode
    validation_loss = 0.0
    with torch.no_grad():  # turn off gradients for validation, saves memory and computations
        total_precision = 0
        for i, (text, bound, mask, input2, labels, heats) in progress_bar:
            text, bound, mask, input2, labels, heats = Variable(text).to(device), Variable(bound).to(device), Variable(mask).to(device), Variable(input2).to(device), Variable(labels).to(device), Variable(heats).to(device)            

            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            if decoder_name == "point":
                labels = heats
            precision = check_clicks(outputs, labels, decoder_name)

            # accumulate validation loss
            validation_loss += loss.item()
            total_precision += precision

    validation_loss = validation_loss / len(data_loader)  # get average loss
    validation_precision = total_precision / len(data_loader)
    progress_bar.close()
    return validation_loss, validation_precision

def train(args):
    # Configuration
    num_epochs = args.epochs
    batch_size = args.batch_size
    val_batch_size = args.val_batch_size
    learning_rate = args.lr
    dataset_path = args.dataset_path
    model_path = args.model_path
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
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    valid_dataset = Dataset(data_dir=dataset_path, type="val", csv_file=csv_path, demo=test, decode_type=decoder_name)
    valid_data_loader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    test_dataset = Dataset(data_dir=dataset_path, type="test", csv_file=csv_path, demo=test, decode_type=decoder_name)
    test_data_loader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False, num_workers=4)

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using {device} device")
    model = model.to(device)

    # Define loss function and optimizer
    if decoder_name == "heatmap":
        criterion = FocalLoss(loss_alpha, loss_gamma)
    else:
        criterion = torch.nn.MSELoss()
    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    # open the csv file in write mode
    import csv

    with open(f'{save_path}/losses,{loss_alpha},{loss_gamma}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # write the header
        writer.writerow(["Epoch", "Training Loss", "Validation Loss", "Valid Precision", "Test Precision"])

    torch.save(model.state_dict(), f'{save_path}/model_{0}.pth')

    early_stopper = EarlyStopper(patience=6, min_delta=0.00001)

    for epoch in range(num_epochs):
        # Instantiate a progress bar object with the total length equal to the size of the data loader
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        
        training_loss = 0.0

        for i, (text, bound, mask, input2, labels, heats) in progress_bar:
            text, bound, mask, input2, labels = Variable(text).to(device), Variable(bound).to(device), Variable(mask).to(device), Variable(input2).to(device), Variable(labels).to(device)

            # print the type of the input
        
            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            
            # if loss.item() == nan or math.isnan(loss.item()):
            #     continue
            # Backward and optimize

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # accumulate training loss
            training_loss += loss.item()
            # print(training_loss)

            # Update the progress bar
            progress_bar.set_description(f'Epoch: {epoch+1}/{num_epochs}')

            # print loss every 100 iterations
            # if i % 100 == 0:
            #     print(f'Epoch: {epoch+1}/{num_epochs}, Iteration: {i+1}/{len(train_data_loader)}, Loss: {loss.item():.4f}, Training Loss: {training_loss:.4f}')

        
        total_training_loss = training_loss
        training_loss = training_loss / len(train_data_loader)  # get average loss

        # After one epoch, close the progress bar
        progress_bar.close()
        torch.save(model.state_dict(), f'{save_path}/model_{epoch+1}.pth')
        # Checkpoint saving example for extracting sub-models
        if hasattr(model, 'laModel'):
            torch.save(model.laModel.state_dict(), os.path.join(save_path, "LAModel", f'LAModel_checkpoint{epoch+1}.pth'))

        
        # Validate the model
        validation_loss, precision = validate(model, valid_data_loader, criterion, device, decoder_name)
        test_loss, test_precision = validate(model, test_data_loader, criterion, device, decoder_name)

        print(f'Epoch: {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Valid Precision: {precision:.4f}, Test Precision: {test_precision:.4f}')
        model.train()  # set the model back to training mode

        # Write the losses to the csv file
        with open(f'{save_path}/losses,{loss_alpha},{loss_gamma}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, total_training_loss, validation_loss, precision, test_precision])
        
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
    args = parser.parse_args()
    train(args)
