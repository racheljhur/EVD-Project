# copied data loader file from a general-purpose script
# author: Rachel Hur | contact: jhur64@gatech.edu
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import argparse

# code usage: python data.py --data-dir data --train-fraction 0.80 --pc-scores-only --model-type cnf

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type = str, help = 'data directory containing processed ml datasets.')
    parser.add_argument('--train-fraction', type = float)
    parser.add_argument('--pc-scores-only', action = 'store_true')
    parser.add_argument('--model-type', type = str)
    parser.add_argument('--glob', action = 'store_true')
    parser.add_argument('--loc', action = 'store_true')

    return parser.parse_args()

def data_loader(dataDir: str) -> NDArray:
    pc_scores_path = os.path.join(dataDir, 'tiled_pc_scores.npy')
    exceed_vals_path = os.path.join(dataDir, 'flattened_exc_vals.npy')

    pc_scores, exceed_vals = np.load(pc_scores_path), np.load(exceed_vals_path)

    return pc_scores, exceed_vals

def getCnfData(data_dir: str, train_fraction: float,
               numRves: int, numSves:int, 
               numExceed: int, pc_scores_only: bool,
               glob: bool):

    # i need to move datasets onto cuda before loading them onto train loader.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairedXLoc, _, pairedPcSc, pairedExceed = data_loader(data_dir) # these are of shape (15, 11400, dim)

    if pc_scores_only:
        print('returning features only with pc scores.')
        scaler = Scaler(pairedPcSc, pairedExceed)
        numRves, numSves, numExceed = scaler.numRves, scaler.numSves, scaler.numExceed
        input_scaler, output_scaler = scaler.get_input_scaler(), scaler.get_output_scaler()
        x_scaled = input_scaler.fit_transform(pairedPcSc.reshape(numRves * numSves * numExceed, pairedPcSc.shape[-1]))
        y_scaled = output_scaler.fit_transform(pairedExceed.reshape(numRves * numSves * numExceed, 1))

    x_scaled = x_scaled.reshape(numRves, numSves * numExceed, x_scaled.shape[-1])
    y_scaled = y_scaled.reshape(numRves, numSves * numExceed,) # this should be fed in as (n,) - i cant remember

    print(x_scaled.shape), print(y_scaled.shape) # (15, 11400, dim)

    if train_fraction == 0.80:
        np.random.seed(0)
        numRves = [i for i in range(15)]
        numRves_copy = numRves.copy()
        testIdx = np.random.choice(numRves, size = 3, replace = False)

        testIdx = [8 10 12] # to match the values used in the paper.

        # this is my rve-level 80-20 split.
        # sorry to alan turing for this sinful code block
        trainIdx = [idx for idx in numRves if idx != testIdx[0]]
        trainIdx = [idx for idx in trainIdx if idx != testIdx[1]]
        trainIdx = [idx for idx in trainIdx if idx != testIdx[2]]

        print(testIdx), print(trainIdx)

    x_train, y_train = x_scaled[trainIdx], y_scaled[trainIdx]
    x_test, y_test = x_scaled[testIdx], y_scaled[testIdx]

    x_train = torch.as_tensor(x_train.reshape(len(trainIdx) * numSves * numExceed, x_train.shape[-1]), dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train.reshape(len(trainIdx) * numSves * numExceed,), dtype=torch.float32, device=device)

    x_test = torch.as_tensor(x_test.reshape(len(testIdx) * numSves * numExceed, x_test.shape[-1]), dtype=torch.float32, device='cpu')
    y_test = torch.as_tensor(y_test.reshape(len(testIdx) * numSves * numExceed,), dtype=torch.float32, device='cpu')

    # zuko flows expect 2 dimensions
    y_test = y_test.unsqueeze(-1)
    y_train = y_train.unsqueeze(-1)

    print(f'\n#-------------------------------')
    print(f'# checking shapes')
    print('#'), print(x_train.shape), print(y_train.shape)
    print(f'#-------------------------------\n')

    n_train = len(x_train)
    torch.manual_seed(42)
    perm = torch.randperm(n_train)

    x_train_shuffled = x_train[perm]
    y_train_shuffled = y_train[perm]

    # i dont think i ever use this
    inverse_perm = torch.argsort(perm)

    # making this copy for evaluation.
    train_dataset_unshuffled = TensorDataset(x_train, y_train)

    train_dataset = TensorDataset(x_train_shuffled, y_train_shuffled)
    test_dataset = TensorDataset(x_test, y_test)

    batch_size = 8192

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    eval_loader = DataLoader(
            train_dataset_unshuffled,
            batch_size = n_train,
            shuffle=False
            )

    return train_loader, test_loader, eval_loader, inverse_perm


if __name__ == "__main__":
    args = parse_args()
    dataDir = args.data_dir
    trainFrac = args.train_fraction
    numRves, numSves, numExceed = 15, 200, 57

    train_loader, test_loader, _, _ = getCnfData(args.data_dir, trainFrac, numRves,
                                                     numSves, numExceed, args.pc_scores_only, args.glob)

