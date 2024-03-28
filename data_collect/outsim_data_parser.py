import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Creates a Dataset for a 6 DOF State. 
# Controls 3DOF ['bhrottle', 'brake', 'steering']
class OutsimDataParser_6X_3U(Dataset):
    def __init__(self, csv_file, name):
        self.read_csv(csv_file)
        self.parse_data()
        self.data_name = name

    def __len__(self):
        return len(self.state_and_control)

    def __getitem__(self, idx):
        xu = self.state_and_control.iloc[[idx]]
        nX = self.next_state.iloc[[idx]]
        return xu.to_numpy(), nX.to_numpy()

    def read_csv(self, data_csv):
        self.data = pd.read_csv(data_csv)

    def plot_trajectory(self):
        self.data.plot(title=self.data_name, kind = 'line', x = 'pos_x', y='pos_y')
        plt.show()

    def parse_data(self):
        self.state_and_control = self.data[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'throttle', 'brake', 'steering']].iloc[1:]
        self.next_state = self.data[['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z']].iloc[:-1]
