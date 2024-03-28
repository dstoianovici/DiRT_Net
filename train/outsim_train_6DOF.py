import os
import sys
sys.path.append('./')
from data_collect.outsim_data_parser import OutsimDataParser_6X_3U
import torch
from torch import nn
from torch.utils.data import DataLoader
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
        print(f"Using Device: {self.device}")

        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9*9, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6),
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




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
    print(f"Training next_data batch shape: {train_next_state.size()}")
    valid_state, valid_next_state = next(iter(validation_data_loader))
    print(f"Validation state_and_control batch shape: {valid_state.size()}")
    print(f"Validation next_data batch shape: {valid_next_state.size()}")
    

    #Instantiate Training Class
    model = Train6DOF_Net()
    model.to(model.device)
    print(model)

if __name__ == "__main__":
    main() 