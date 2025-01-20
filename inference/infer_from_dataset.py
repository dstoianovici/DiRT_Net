import os, sys
import torch
from torch.utils.data import Dataset
import torch
sys.path.append('./')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train.outsim_train_6DOF import Train6DOF_Net

sys.path.append('./')
from data_collect.outsim_data_parser import OutsimDataParser_6X_3U
from data_collect.outsim_data_collect import OutSim_Data_Collect
from train.outsim_train_6DOF import Train6DOF_Net

model_path = "models\model_20250106_201842_10"

model = torch.load(model_path)

print(model)
