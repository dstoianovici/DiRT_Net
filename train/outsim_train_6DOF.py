import os
import sys
from datetime import datetime
sys.path.append('./')
from data_collect.outsim_data_parser import OutsimDataParser_6X_3U
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt

class Train6DOF_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.to(self.device)
        print(f"Using Device: {self.device}")

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9,6),
            nn.ReLU(),
            nn.Linear(6, 6),
            nn.ReLU(),
            nn.Linear(6, 6)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train_one_epoch(epoch_index, tb_writer, data_loader, loss_fn, optimizer, model):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(data_loader):
        state, next_state = data
        optimizer.zero_grad()
        next_state_pred = model(state)
        loss = loss_fn(next_state_pred, next_state)
        loss.backward()
        optimizer.step()

        running_loss+= loss.item()
        if i % 1000 == 999:
            last_loss = running_loss/1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def main():
    #Get Training Data and Visualize
    training_data = OutsimDataParser_6X_3U("data\OutSim_13-Feb-24-21-21-46.csv", "Training Data")
    validation_data = OutsimDataParser_6X_3U("data\OutSim_13-Feb-24-21-31-19.csv", "Validation Data")
    # training_data.plot_trajectory()
    # validation_data.plot_trajectory()

    #Create Data Loaders
    training_data_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    validation_data_loader = DataLoader(validation_data, batch_size=64, shuffle=True)

    #Inspect Loaded Data
    train_state, train_next_state = next(iter(training_data_loader))
    print(f"Training state_and_control batch shape: {train_state.size()}")
    print(f"Training data datatype: {train_state.dtype}")
    print(f"Training next_data batch shape: {train_next_state.size()}")
    valid_state, valid_next_state = next(iter(validation_data_loader))
    print(f"Validation state_and_control batch shape: {valid_state.size()}")
    print(f"Validation data datatype: {valid_state.dtype}")
    print(f"Validation next_data batch shape: {valid_next_state.size()}")
    

    #Instantiate Training Class
    model = Train6DOF_Net()
    print(f"Model:\n{model}")



    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/dirtAI_trainer_6x3u_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 25

    best_vloss = 1_000_000.

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0
    last_loss = 0

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, writer, training_data_loader, loss_fn, optimizer, model)


        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_data_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

if __name__ == "__main__":
    main() 