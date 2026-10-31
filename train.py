# copied training file from a general-purpose script
# author: Rachel Hur | contact: jhur64@gatech.edu
import os
import time
import argparse
from datetime import date
import random

import numpy as np
from numpy.typing import NDArray

import tqdm
import ot as pot
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchdyn
import torchsde
from torchdyn.core import NeuralODE
from torchdiffeq import odeint
import zuko
import json

import gpytorch

from classes import *
from data import getCnfData

# you only need to run this and eval.py
# usage: python train.py --ver 0 --model-type cnf --data-dir data --pc-scores-only --num-epochs 250 --batch-size 8192

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', type = int, help = 'training version.')
    parser.add_argument('--model-type', type = str)
    parser.add_argument('--data-dir', type = str)
    parser.add_argument('--train-fraction', type = float, default = 0.8)
    parser.add_argument('--pc-scores-only', action = 'store_true')
    parser.add_argument('--glob', action = 'store_true')
    parser.add_argument('--num-epochs', type = int)
    parser.add_argument('--batch-size', type = int)

    return parser.parse_args()

def load_model(model_type, dim, cdim,
               x_train: None | torch.Tensor,
               y_train: None | torch.Tensor):

    # for constant weight initialization across versions.
    torch.manual_seed(42)

    if model_type == 'cnf':
        from flows import CNF # this version has a modified mlp with dropout
        return CNF(features=dim, context=cdim, hidden_features=(256, 256, 256), activation=nn.ELU, p_drop = 0.0)
    #elif model_type == 'gpr':
    #    assert x_train is not None, 'you must load in training datasets'
    #    return gpytorch.likelihoods.GaussianLikelihood(), SparseGPModel() # by default, inducing_points = :500

    raise ValueError(f'unknown model_type: {model_type}')

def load_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr)

def load_scheduler(model, optimizer, trainIter):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=trainIter)

def train_loop(x_train: None | torch.Tensor, y_train: None | torch.Tensor,
        train_loader: None | torch.Tensor,
        likelihood: None | nn.Module,
        model: nn.Module, model_type, params: tuple,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: str | torch.device,
        write_path: str) -> tuple[None, NDArray]:

    if model_type == 'cnf':
        dim, cdim, batch_size, num_t, trainIter = params
        z_train = []
        iter_losses = []

        for i in range(trainIter):
            losses = []
            for x, y in train_loader: # this is loaded as (pcSc, extremeVals)

                optimizer.zero_grad()
                loss = -model(x).log_prob(y).mean()
                loss.backward()
                optimizer.step()
                losses.append(loss.detach())
                iter_losses.append(torch.stack(losses).mean().item())
                # Append relevant data for validation at the last training iteration to avoid memory accumulation
                if i == trainIter - 1:
                    with torch.no_grad():
                        z_train.append(model.transform(x)(y).cpu())

                if (i + 1) % 25 == 0:
                    print(f"{i+1}: loss {iter_losses[-1]:0.3f}")

        z_train = torch.cat(z_train, dim=0).numpy()

        return torch.save(model.state_dict(), write_path), np.array(iter_losses), z_train

    raise ValueError(f'unknown model_type: {model_type}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parse_args()

    d = date.today()

    write_dir = os.path.join('cnfTrainOut', f'{d}_{args.ver}')
    os.makedirs(write_dir, exist_ok=True)

    # this function returns NDArrays.
    train_loader, _, _, _ = getCnfData(data_dir=args.data_dir, train_fraction=args.train_fraction,
                numRves=15, numSves=200, numExceed=57, pc_scores_only=args.pc_scores_only, glob=args.glob)

    #----------------------------------------------------
    # main training block
    config = {
        'date': f"{d}",
        'ver': args.ver,
        'model': f'{args.model_type}',
        'optim': 'Adam',
        'numIter': args.num_epochs,
        'numHidden': 3,
        'w size': "256,256,256",
        'dropout': 0.0,
        'batch_size': 8192,
        }
    if args.pc_scores_only:
        config['notes'] = "this version only used pc scores of fiber autocorr"
    if args.glob:
        config['notes'] = "added cross correlations."
    else:
        config['notes'] = "this version used both pc scores of fiber autocorr and cld (x) of the matrix phase."

    config['extra notes'] = "the test set was [8 10 12]"

    metaDataPath = write_dir
    with open(metaDataPath + f"/{args.model_type}.json", "w") as f:
        json.dump(config, f, indent=2)

    y_tensor = train_loader.dataset.tensors[1]
    dim = 1 if y_tensor.ndim == 1 else y_tensor.shape[-1]
    cdim = train_loader.dataset.tensors[0].shape[-1]

    model = load_model(args.model_type, dim, cdim, None, None).to(device)
    optimizer = load_optimizer(model, lr = 1e-3)

    batch_size, num_t, trainIter = 8192, 10, args.num_epochs

    params = (dim, cdim, batch_size, num_t, trainIter)
    # will auto save the trained model.
    _, losses, z_learned = train_loop(None, None, train_loader,
                                      None, model, args.model_type,
                                      params, optimizer, None,
                                      device, os.path.join(write_dir, 'model.pth'))
    np.save(os.path.join(write_dir, 'losses.npy'), losses)
    np.save(os.path.join(write_dir, 'z_learned.npy'), z_learned)
    #------------------------------------------------------

    print(f"\nFinished training and writing results.")


~
~
~
~
~
