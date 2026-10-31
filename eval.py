# copied from a general-purpose evaluation code.
# author: Rachel Hur | contact: jhur64@gatech.edu
import numpy as np
import torch
import gpytorch
import os
import argparse
from classes import *
import time
import ot as pot

import zuko
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt

# usage: python eval.py --ver 0 --model-type cnf --data-dir data --train-fraction 0.80
#        \ --pc-scores-only --num-epochs 300 --batch-size 8192 --write-dir cnfEvalOut

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ver', type = int, help = 'training version.')
    parser.add_argument('--model-type', type = str)
    parser.add_argument('--data-dir', type = str)
    parser.add_argument('--train-fraction', type = float, default = 0.8)
    parser.add_argument('--pc-scores-only', action = 'store_true')
    parser.add_argument('--glob', action = 'store_true') # unused

    parser.add_argument('--num-epochs', type = int)
    parser.add_argument('--write-dir', type = str)
    parser.add_argument('--batch-size', type = int)
    parser.add_argument('--num-trials', type = int) # unused

    return parser.parse_args()

def import_modules(model_type: str) -> None:
    if model_type == 'otcfm':
        from data import getOtCfmData
        from data import Scaler

def load_model(model_type, dim, cdim,
               x_train: None | torch.Tensor,
               y_train: None | torch.Tensor):
    if model_type == 'cnf':
        from flows import CNF # this version has a modified mlp with dropout
        # model settings are set to the parameters listed in the published paper.
        return CNF(features=dim, context=cdim, hidden_features=(256,256,256), activation=nn.ReLU, p_drop = 0.0)

    raise ValueError(f'unknown model_type: {model_type}')

def load_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr)

def visualizeLoss(loss, figurePath):
    x = np.linspace(0, len(loss), len(loss))
    plt.figure()
    plt.plot(x,loss)
    plt.xlabel('iter')
    plt.ylabel('nll')
    plt.savefig(os.path.join(figurePath, 'loss_prog.png'))
    plt.close()

def visualizeParity(pred, grTruth, nmae, rmse, figurePath, train = False):
    plt.figure()
    plt.scatter(grTruth, pred, label=f"GP Predictions\nrmse = {rmse:.3f}\nnmae = {nmae:.3f}")
    plt.plot(grTruth, grTruth, color="red", label="Perfect Parity")
    plt.xlabel("GP-Predicted extremeVals (MPa)")
    plt.ylabel("FE-Computed extremeVals (MPa)")
    plt.legend()
    if not train:
        plt.title(f"Test Parity all SVEs")
        plt.savefig(os.path.join(figurePath, f"parity_sve_level_test.png"))
    else:
        plt.title(f"Train Parity all SVEs")
        plt.savefig(os.path.join(figurePath, f"parity_sve_level_train.png"))

    plt.close()

def getFitnessMetrics(modelPrediction, grTruth):
    errors = modelPrediction - grTruth

    rmse = np.sqrt(np.mean(errors ** 2))
    nmae = np.mean(np.abs(errors)) / (grTruth.max() - grTruth.min())

    mean = np.mean(grTruth, axis = 0)[0]

    print(f'\nnmae should be <1.00.')
    print(f'\nrmse should be 2 orders of magnitude less than the nominal predictive quantity.')
    print(f'\n nominal predictive quantity: {mean:.3f}')

    print('\nPERFORMANCE METRICS (USING ORIGINAL VALUES)')
    print(f'nmae = {nmae:.3f}')
    print(f'rmse = {rmse:.3f}')

    return nmae, rmse

def eval_cnf(model, model_state_dir, loader, device='cpu', train=False):

    model.load_state_dict(torch.load(model_state_dir, map_location=device, weights_only=True))
    model.to(device).eval()

    num_samples, num_trials, chunk_size = 1, args.num_trials, 256

    y_true, y_pred, _, nll = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            dist = model(x)

            y_pred.append(dist.sample().cpu())
            y_true.append(y.cpu())
            nll.append(-dist.log_prob(y).cpu())

    y_pred = torch.cat(y_pred).float()
    y_true = torch.cat(y_true).float()
    nll = torch.cat(nll)

    print(f"mean NLL: {nll.mean():.3f}")
    return y_pred, y_true, nll.numpy()

def plot_evds(y1_predicted, y_true, bins, write_dir, rveNum, d, ver):

    # compute wasserstein-1 distance
    y1_predicted = np.asarray(y1_predicted).ravel()
    y_true = np.asarray(y_true).ravel()
    w1 = wasserstein_distance(y1_predicted, y_true) # by default this is w-1

    edges = np.histogram_bin_edges(np.concatenate([y1_predicted, y_true]), bins=30)

    plt.hist(y_true, bins = edges, color = 'cyan', alpha = 0.75,  edgecolor = 'black', label = f'fea', density = True)
    plt.hist(y1_predicted, bins = edges, color = 'purple', alpha = 0.75, edgecolor = 'black', label = f'{args.model_type}\nw1={w1:.3f}', density = True)
    plt.legend()
    plt.xlabel('max principal stress (MPa)')
    plt.ylabel(r'probability density $p_\theta(y \mid \{m\})$')
    #plt.xlim(1.25, 3.50)
    plt.savefig(os.path.join(write_dir, f'{d}_{ver}', f'exceedDists_rve_{rveNum}.png'), dpi = 300) # dpi is high for the manuscript figures.
    plt.close()

    # because i am not using the keyboard shortcut for mass commenting in vi
    if False:
        plt.figure()
        plt.hist(y_true, bins = edges, color = 'blue', alpha = 0.75,  edgecolor = 'black', label = f'fea', density = True)
        plt.legend()
        plt.xlabel('max principal stress (MPa)')
        plt.ylabel(r'probability density $p_\theta(y \mid \{m\})$')
        plt.savefig(os.path.join(write_dir, f'{d}_{ver}', f'trueDistsOnly_rve_{rveNum}.png'), dpi = 300)
        plt.close()

def plot_latent_qq(z, write_dir, d, ver):

    z = np.asarray(z).flatten()

    plt.figure()
    probplot(z, dist='norm', plot=plt)
    plt.xlabel('Theoretical Quantiles for the Unit Normal', fontsize=16)
    plt.ylabel('Sample Quantiles', fontsize=16)
    plt.title('Latent QQ Plot', fontsize=16)
    plt.grid(True, alpha=0.4)
    #plt.tight_layout()
    plt.savefig(os.path.join(write_dir, f'{d}_{ver}', f'z_learned_qq.png'), dpi = 300)
    plt.close()


def plot_moment_err(err_mean_per, write_dir, d, ver):

    rve_nums = sorted(err_mean_per.keys(), key=int)
    mean_err = np.array([err_mean_per[k] for k in rve_nums]) * 100

    markers = {'1': 's', '6': 's', '8': 's'}

    plt.figure()
    for rve, err in zip(rve_nums, mean_err):
        plt.scatter(
            int(rve), err,
            color='red' if rve in markers else 'blue',
            marker=markers.get(rve, 'o'))

    plt.title('Percent Error in Moment Estimates', fontsize=18)
    plt.xlabel('RVE', fontsize=16)
    plt.ylabel('Percent Error (%)', fontsize=16)
    plt.xticks([int(k) for k in rve_nums], [int(k) + 1 for k in rve_nums])
    plt.grid(True, alpha=0.4)
    #plt.legend()
    #plt.tight_layout()

    save_dir = os.path.join(write_dir, f'{d}_{ver}')
    os.makedirs(save_dir, exist_ok=True)

    plt.savefig(
        os.path.join(save_dir, 'stats_err_rve.png'),
        dpi=300
    )
    plt.close()



if __name__ == "__main__":
    args = parse_args()
    import_modules(args.model_type) # exposes required methods

    if args.pc_scores_only:
        pairedX = np.load(os.path.join(args.data_dir, 'tiled_pc_scores.npy'))
        exceedVals = np.load(os.path.join(args.data_dir, 'flattened_exc_vals.npy'))

    rveNumsTest = [8, 10, 12]
    rveNumsTrain = [0, 1, 2, 3, 4, 5, 7, 9, 11, 13, 14]


    #------------------------------------------------------------------
    # helper constructor
    # only running fit() to create the fitted scaler instance.
    scaler = Scaler(pairedX, exceedVals)
    numRves, numSves, numExceed = scaler.numRves, scaler.numSves, scaler.numExceed
    input_scaler, output_scaler = scaler.get_input_scaler(), scaler.get_output_scaler()
    # important! use the entire original datasets to scale.

    input_scaler.fit(pairedX.reshape(numRves * numSves * numExceed, pairedX.shape[-1]))
    output_scaler.fit(exceedVals.reshape(numRves * numSves * numExceed, 1))
    #------------------------------------------------------------------

    #-------------------------------------------------------------------
    # main evaluation block
    #
    d = '2026-10-31'

    os.makedirs(f'{args.write_dir}/{d}_{args.ver}', exist_ok = True)

    if args.model_type == 'cnf':
        _, test_loader, eval_loader, inverse_perm = getCnfData(data_dir=args.data_dir, train_fraction=args.train_fraction,
                numRves=15, numSves=200, numExceed=56, pc_scores_only=args.pc_scores_only, glob=args.glob)
        y_tensor = eval_loader.dataset.tensors[1]

        #--------------------------------------------------------
        # check normality
        z = np.load(f'cnfTrainOut/{d}_{args.ver}/z_learned.npy')
        plot_latent_qq(z, args.write_dir, d, args.ver)
        #--------------------------------------------------------

        dim = 1 if y_tensor.ndim == 1 else y_tensor.shape[-1]
        cdim = eval_loader.dataset.tensors[0].shape[-1]

        model = load_model(args.model_type, dim, cdim, x_train = None, y_train = None).to(device)
        optimizer = load_optimizer(model, lr = 1e-3)

        #---------------------------------------------------------------------------------
        # prediction block
        sigma, batch_size, num_t = 1e-3, args.batch_size, 10
        model_state_dir = os.path.join('cnfTrainOut', f'{d}_{args.ver}', 'model.pth')
        os.makedirs(os.path.join(args.write_dir, f'{d}_{args.ver}'), exist_ok = True)

        y_predicted_train, y_train, y_ext_train, nll_train = eval_cnf(model, model_state_dir, eval_loader, train=True)
        y_predicted_test,  y_test, y_ext_test,  nll_test  = eval_cnf(model, model_state_dir, test_loader)
        #---------------------------------------------------------------------------------
    #-------------------------------------------------------------------



    #--------------------------------------------------------------------
    # evaluation metrics
    #

    elif args.model_type == 'cnf':
        lossProgPath = os.path.join('cnfTrainOut', f'{d}_{args.ver}', 'losses.npy')
        losses = np.load(lossProgPath)

        y_predicted_train, y_train = output_scaler.inverse_transform(y_predicted_train), output_scaler.inverse_transform(y_train)
        y_ext_train = output_scaler.inverse_transform(y_ext_train)

        y_predicted_test, y_test = output_scaler.inverse_transform(y_predicted_test), output_scaler.inverse_transform(y_test)
        y_ext_test = output_scaler.inverse_transform(y_ext_test)

    if args.model_type == 'otcfm' or args.model_type == 'cnf':

        y_predicted_train = y_predicted_train.reshape(len(rveNumsTrain), 200 * 56, 1)
        y_predicted_test = y_predicted_test.reshape(len(rveNumsTest), 200 * 56, 1)

        y_train = y_train.reshape(len(rveNumsTrain), 200 * 56, 1)
        y_test = y_test.reshape(len(rveNumsTest), 200 * 56, 1)

        visualizeLoss(losses, os.path.join(args.write_dir,f'{d}_{args.ver}'))

        exceedValsTrain = exceedVals[rveNumsTrain]
        #y_predicted_train = y_predicted_train.reshape(12, 56 * 200, 1)
        y_ext_train = y_ext_train.reshape(12, 56 * 200, 1)
        y_ext_train = y_ext_train.mean(axis = 1) # average across all sves for the extreme-most value per rve.

        per_err_mean = {}
        per_err_std = {}
        for i, rvenum in enumerate(rveNumsTrain):
            y_pred = y_predicted_train[i]
            y_true = exceedValsTrain[i]
            y_ext = y_ext_train[i]
            plot_evds(y_pred, y_true, bins, args.write_dir, rvenum, d, args.ver)

            y_pred_mean, y_true_mean = float(y_pred.mean()), float(y_true.mean())
            y_pred_std,  y_true_std  = float(y_pred.std()),  float(y_true.std())
            y_pred_ext,  y_true_ext  = float(y_ext.max()),   float(y_true.max())

            per_err_mean[f'{rvenum}'] = np.abs(y_pred_mean - y_true_mean) / y_true_mean
            per_err_std[f'{rvenum}'] = np.abs(y_pred_std - y_true_std) / y_true_std

        exceedValsTest = exceedVals[rveNumsTest]
        y_predicted_test = y_predicted_test.reshape(3, 56 * 200, 1)
        for i, rvenum in enumerate(rveNumsTest):
            y_pred = y_predicted_test[i]
            y_true = exceedValsTest[i]
            y_ext = y_ext_test[i]
            plot_evds(y_pred, y_true, bins, args.write_dir, rvenum, d, args.ver)

            y_pred_mean, y_true_mean = float(y_pred.mean()), float(y_true.mean())
            y_pred_std,  y_true_std  = float(y_pred.std()),  float(y_true.std())
            y_pred_ext,  y_true_ext  = float(y_ext.max()),   float(y_true.max())

            per_err_mean[f'{rvenum}'] = np.abs(y_pred_mean - y_true_mean) / y_true_mean
            per_err_std[f'{rvenum}'] = np.abs(y_pred_std - y_true_std) / y_true_std


        #plot_moment_err(per_err_mean, args.write_dir, d, args.ver)

