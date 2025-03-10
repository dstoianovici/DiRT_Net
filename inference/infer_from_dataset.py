import os, sys
import torch
from torch.utils.data import Dataset
import torch
sys.path.append('./')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_collect.outsim_data_parser import OutsimDataParser_6X_3U
from data_collect.outsim_data_collect import OutSim_Data_Collect
from train.outsim_train_6DOF import Train6DOF_Net


# Load Model Weights
# model_path = "models\model_20250106_201238_195" # BEST SO FAR
model_path = "models\\20250129_150212\\model_64"
model = Train6DOF_Net()
model.load_state_dict(torch.load(model_path))
# model = torch.load(model_path)
model.eval()
# print(model)

# Prepare Training Data (Using Validation Data before  new collect)
# validation_data = OutsimDataParser_6X_3U("data\OutSim_13-Feb-24-21-31-19.csv", "Test Data")
# test_data = OutsimDataParser_6X_3U("data\OutSim_13-Feb-24-21-31-19.csv", "Test Data")
test_data = OutsimDataParser_6X_3U("data\OutSim_24-Jan-25-00-14-17.csv", "Test Data")
input_tensor = torch.tensor(test_data.state_and_control.values)

# Infer on Data
with torch.no_grad():
    output = model(input_tensor)

output_np = output.detach().numpy()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(test_data.next_state['pos_x'], test_data.next_state['pos_y'], color='tab:blue', label="truth")
ax.plot(output_np[:,0], output_np[:,1], color='tab:red', label="inference")
plt.show()
