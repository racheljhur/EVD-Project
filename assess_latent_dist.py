import numpy as np
import torch
from datetime import date
from scipy.stats import norm
import matplotlib.pyplot as plt
import os
from scipy.stats import probplot

# cnfTrainOut/2026-09-16_0/latentOut

d, v = date.today(), 0
d, v = '2026-09-16', 0

zLearnedPath = os.path.join('cnfTrainOut', f'{d}_{v}', 'latentOut', 'z_train.npy')
figPath = os.path.join('cnfTrainOut', f'{d}_{v}', 'latentOut')
zLearned = np.load(zLearnedPath)


def plot_latent_qq(z, write_dir, d, ver):

    z = np.asarray(z).flatten()

    plt.figure()
    probplot(z, dist='norm', plot=plt)
    plt.xlabel('Theoretical Quantiles for the Unit Normal', fontsize=16)
    plt.ylabel('Sample Quantiles', fontsize=16)
    plt.title('Latent QQ Plot', fontsize=16)
    plt.grid(True, alpha=0.4)
    #plt.tight_layout()
    plt.savefig('z_learned_qq.png', dpi = 300)
    plt.close()

cwd = os.getcwd()
plot_latent_qq(zLearned, cwd, d, v)

