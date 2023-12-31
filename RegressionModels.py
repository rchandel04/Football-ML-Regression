############# all imports ################
# datasets
import pandas as pd
import nfl_data_py as nfl
from dataScrapping import generateData
import numpy as np
import matplotlib.pyplot as plt
# models
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
##########################################

# construct linear model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
        
    def forward(self, x):
        return self.linear_stack(x)

#construct non-linear model
class NonLinearRegressionModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units = 8):
        super().__init__()
        self.linear_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features),
        )
        
    def forward(self, x):
        return self.linear_stack(x)
    
