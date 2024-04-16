import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn as nn

from tqdm import tqdm
import argparse
import os 
import csv
import matplotlib.pyplot as plt

from numpy import nan
import math
from tools.loss import FocalLoss
from tools.hp5dataset import MultiFileDataset as Dataset
from tools.hp5dataset import custom_collate_fn
# from tools.hp5dataset import SplitDataset as Dataset
from tools.model import VLModel
from tools.utils import EarlyStopper

Random = 880323

def train(args):
    # Configuration
    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    dataset_path = args.dataset_path
    model_path = args.model_path
    csv_path = args.csv_path
    loss_alpha = args.loss_alpha
    loss_gamma = args.loss_gamma
    save_path = args.save_path
    test = args.test
    save_path = f"{save_path}/{loss_alpha}_{loss_gamma}"

    # fix random seeds for reproducibility
    torch.manual_seed(Random)
    torch.cuda.manual_seed(Random)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    demo = False
    if test:
        demo = True
    # Prepare your data loader
    train_dataset = Dataset(data_dir=dataset_path, train=True, csv=csv_path, demo=demo)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, collate_fn = custom_collate_fn)

    valid_dataset = Dataset(data_dir=dataset_path, train=False, csv=csv_path, demo=demo)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=16, collate_fn = custom_collate_fn)

    # Load your model
    model = VLModel()

    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = FocalLoss(loss_alpha, loss_gamma)
    # criterion = torch.nn.L1Loss()
    # criterion = torch.nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # open the csv file in write mode
    import csv

    # open the csv file in write mode
    if not os.path.exists("loss"):
        os.makedirs("loss")

    with open(f'loss/losses,{loss_alpha},{loss_gamma}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # write the header
        writer.writerow(["Epoch", "Training Loss", "Validation Loss"])

    torch.save(model.state_dict(), f'{save_path}/model_{0}.pth')

    early_stopper = EarlyStopper(patience=2, min_delta=0.00001)

    for epoch in range(num_epochs):
        # Instantiate a progress bar object with the total length equal to the size of the data loader
        progress_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        
        training_loss = 0.0

        for i, (text, bound, mask, input2, labels) in progress_bar:
            text, bound, mask, input2, labels = Variable(text).to(device), Variable(bound).to(device), Variable(mask).to(device), Variable(input2).to(device), Variable(labels).to(device)

            outputs = model(text, bound, mask, input2)
            loss = criterion(outputs, labels)
            
            if loss.item() == nan or math.isnan(loss.item()):
                continue
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

        # Now evaluate on the validation set
        validation_loss = 0.0
        model.eval()  # set the model to evaluation mode

        with torch.no_grad():  # turn off gradients for validation, saves memory and computations
            for text, bound, mask, input2, labels in valid_data_loader:
                text, bound, mask, input2, labels = Variable(text).to(device), Variable(bound).to(device), Variable(mask).to(device), Variable(input2).to(device), Variable(labels).to(device)                

                outputs = model(text, bound, mask, input2)
                loss = criterion(outputs, labels)
                if loss.item() == nan or math.isnan(loss.item()):
                    continue

                # accumulate validation loss
                validation_loss += loss.item()

        total_validation_loss = validation_loss
        validation_loss = validation_loss / len(valid_data_loader)  # get average loss
        print(f'Epoch: {epoch+1}/{num_epochs}, Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}')
        model.train()  # set the model back to training mode


        with open(f'loss/losses,{loss_alpha},{loss_gamma}.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch+1, training_loss, validation_loss])
        
        if early_stopper.early_stop(validation_loss):             
            break
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a UNet model")
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--dataset_path", type=str, default=".", help="path to the dataset")
    parser.add_argument("--model_path", type=str, default=None, help="path to the model")
    parser.add_argument("--csv_path", type=str, default=None, help="path to the csv file")
    parser.add_argument("--loss_alpha", type=float, default=0.25, help="alpha for focal loss")
    parser.add_argument("--loss_gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument("--save_path", type=str, default=".", help="path to save the model")
    parser.add_argument("--test", type=bool, default=False, help="test mode")

    args = parser.parse_args()
    train(args)
