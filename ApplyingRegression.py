############# all imports ################
# datasets
import pandas as pd
import nfl_data_py as nfl
from dataScrapping import generate_data
import numpy as np
import matplotlib.pyplot as plt
# models
import torch
import torch.nn as nn
from RegressionModels import LinearRegressionModel, NonLinearRegressionModel
from torch.utils.data import TensorDataset, DataLoader
##########################################

# device-agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"

# constants for model
BATCH_SIZE = 4
INPUT_COLUMNS = ['ht', 'wt', 'forty', 'bench', 'vertical', 'broad_jump']
OUTPUT_COLUMNS = ["receptions", "rec_yards"]
YEARS = np.arange(2000, 2016, 1)
YEARS = YEARS.tolist()
POSITION = 'RB'
OMITNOSHOWS = True

# generate desired data
pos_stats = generate_data(INPUT_COLUMNS, OUTPUT_COLUMNS, YEARS, POSITION, OMITNOSHOWS)
print(pos_stats.head)

# split data based on input and target
inputs = pos_stats[['pick'] + INPUT_COLUMNS]
targets = pos_stats[OUTPUT_COLUMNS]

# transform data frames into tensors
inputs = torch.tensor(inputs.values, dtype=torch.float32)
targets = torch.tensor(targets.values, dtype=torch.float32)
inputs = inputs.to(device)
targets = targets.to(device)

# create dataset and dataloader from tensors
train_dataset = TensorDataset(inputs, targets)
train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

#instantiate model
model = NonLinearRegressionModel(input_features=7, output_features=2, hidden_units=32).to(device)

# define loss function
loss_fn = nn.L1Loss()

# define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop
def fit(num_epochs, model, loss_fn, optimizer):
    cur_loss = 0.0
    for epoch in range(num_epochs):
        model.train()
        for x, y in train_dataloader:
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cur_loss += loss.item()
        
        if epoch % 10 == 0:
            print('training loss for epoch ', epoch, " : ", cur_loss)
        cur_loss = 0.0
    print('Training has finished.')

# fit the model to the training data
fit(num_epochs=50, model=model, loss_fn=loss_fn, optimizer=optimizer)

# define function to plot a specific input to the model against the target and predicted output of the model to observe influence of different inputs
def plot_input_to_target(input_column, output_data, color, label):
    column_data = pos_stats[input_column]
    column_data = column_data.to_numpy()
    plt.scatter(column_data, output_data, marker='o', s=10, color=color, label=label)
    

# evaluate the model on some test data (here we use rb data from 2016)
# test = np.array([[63, 72, 225, 4.5, 23, 37, 115]])
# test = torch.from_numpy(test)
# test = test.to(device)
# test = test.float()

model.eval()

with torch.inference_mode():
    pred = model(inputs)
    
# split predicted output and target output into x and y coordinates
pred = pred.cpu()
targets = targets.cpu()

# target coordinates
x_target = targets[:, 0]
y_target = targets[:, 1]

# predicted coordinates
x_predicted = pred[:, 0]
y_predicted = pred[:, 1]

# plot the data

plt.title('Scatter Plot')
plt.xlabel('pick')
plt.ylabel('rush_atts')

plot_input_to_target('ht', x_target, color='blue', label='targets')
plot_input_to_target('ht', x_predicted, color='green', label='predicted')

plt.grid(True)
plt.legend()
plt.show()