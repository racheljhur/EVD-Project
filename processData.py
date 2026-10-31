import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torchvision.transforms as T
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PCs = np.load("data/pc_scores.npy")
EVs = np.load("data/responses.npy")

# renaming because I need the unprocessed datasets later
gr_truth = EVs
PCs_no_repeat = PCs

# checking the shapes of imported data
print('checking the shapes of imported data')
print(PCs.shape)
print(EVs.shape)

ranges_dict = {
    'F_0.3': (0, 199),
    'FA_0.3': (200, 399),
    'FAAA_0.3': (400, 599),
    'FC_0.3': (600, 766),
    'FCCC_0.3': (767, 966),
    'F_0.4': (967, 1166),
    'FA_0.4': (1167, 1366),
    'FAAA_0.4': (1367, 1610),
    'FC_0.4': (1611, 1810),
    'FCCC_0.4': (1811, 2010),
    'F_0.5': (2011, 2199),
    'FA_0.5': (2200, 2399),
    'FAAA_0.5': (2400, 2599),
    'FC_0.5': (2600, 2799),
    'FCCC_0.5': (2800, 2999)
}

ranges_dict = {
    label: (start * 56, (end + 1) * 56 - 1) for label, (start, end) in ranges_dict.items()
}

# reshape and flatten extreme values
M = EVs.shape[0] * EVs.shape[1]
EVs = EVs.reshape(M,1) # shape (168000, 1)

PCs = np.repeat(PCs, 56, axis=0)[:,:10] # shape (168000, 10)

print('reshaped dataset shapes:')
print(EVs.shape)
print(PCs.shape)

# standardize
input_scaler = StandardScaler()
PCs_scaled = input_scaler.fit_transform(PCs)
PCs_no_rep_scaled = input_scaler.fit_transform(PCs_no_repeat)[:,:10]
data_mean = torch.tensor(input_scaler.mean_, dtype=torch.float32).to(device)
data_std = torch.tensor(input_scaler.scale_, dtype=torch.float32).to(device)

data_mean_np = data_mean.cpu().numpy()
data_std_np = data_std.cpu().numpy()

output_scaler = StandardScaler()
EVs_scaled = output_scaler.fit_transform(EVs)
data_mean_evs = torch.tensor(output_scaler.mean_, dtype=torch.float32).to(device)
data_std_evs = torch.tensor(output_scaler.scale_, dtype=torch.float32).to(device)

data_mean_evs_np = data_mean_evs.cpu().numpy()
data_std_evs_np = data_std_evs.cpu().numpy()

# unscaling functions
def unstandardize_x(x):
    '''go from scaled data x back to the original range'''
    return x * data_std_evs_np + data_mean_evs_np

def unstandardize_y(y):
    '''go from scaled data y back to the original range'''
    return y * data_y_std + data_y_mean

# Split the dataset into test (three classes) and train (remaining 12 classes)
# I want to give the model a hard time. Let's take the three classes that had
# the worst performance when we trained on all classes: FAAA40, FCCC40, and FA50 

test_classes = ['FCCC_0.4', 'FAAA_0.4', 'FA_0.5']
test_indices = np.concatenate([np.arange(start, end + 1) for cls in test_classes for start, end in [ranges_dict[cls]]])

# test_labels = []
# Include this in the class assignment script.
# for test_class in test_classes:
#     start, end = ranges_dict[test_class]
#     test_labels.extend([test_class] * (end - start + 1))  # repeat the label for the index range
# test_labels = np.array(test_labels)

# Set the remaining 3 RVEs to be the test set
train_indices = np.setdiff1d(np.arange(EVs_scaled.shape[0]), test_indices)

y_test = torch.tensor(EVs_scaled[test_indices]).float().to(device)
x_test = torch.tensor(PCs_scaled[test_indices]).float().to(device)

y_train = torch.tensor(EVs_scaled[train_indices]).float().to(device)
x_train = torch.tensor(PCs_scaled[train_indices]).float().to(device)

print(f"Training set shape: {x_train.shape}, {y_train.shape}")
print(f"Test set shape: {x_test.shape}, {y_test.shape}")

# Shape ((131936, 1), (131936, 10))
train_dataset = TensorDataset(x_train, y_train) 

# Shape ((36064, 1), (36064, 10))
test_dataset = TensorDataset(x_test, y_test) 

batch_size = 8192
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

print('Data fully processed.')
