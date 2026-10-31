'''core cnf training script.
'''
import numpy as np
import torch
import zuko
from scipy.stats import linregress
from processData import *
#from classes import *
#from classesSveGen import *
#from priorSampler import *
from datetime import date
import subprocess
import os

d, v = date.today(), 0

outDir = 'cnfTrainOut'
cnfTrainOutDir = os.path.join(outDir,f'{d}_{v}')
os.makedirs(cnfTrainOutDir, exist_ok = True)

latentDistDir = os.path.join(outDir,f'{d}_{v}','latentOut')
os.makedirs(latentDistDir, exist_ok = True)

dim = EVs.shape[-1]
cdim = PCs.shape[-1]

print(dim), print(cdim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

flow = zuko.flows.continuous.CNF(features=dim, context=cdim, hidden_features=(256, 256, 256))
flow.to(device)

optimizer = torch.optim.Adam(flow.parameters(), lr=1e-3, weight_decay=1e-5)

z_train = []
iter_losses = []

n_iters = 300

print("\nTraining...\n",flush=True)
for i in range(n_iters):
    losses = []
    for x, y in train_loader: # this is loaded as (pcSc, extremeVals)

        optimizer.zero_grad()
        loss = -flow(x).log_prob(y).mean()
        loss.backward()
        optimizer.step()
        
        # This is to save the loss per batch, per training iteration
        #losses.append(loss.detach())

        # Append relevant data for validation at the last training iteration to avoid memory accumulation
        if i==n_iters-1:
            # This will append across all batches at the final training iteration
            z_learned = flow.transform(x)(y).detach()
            z_train.append(z_learned)


    # Average loss across all 20 batches per iteration
    #avg_loss = torch.stack(losses).mean().item()
    #iter_losses.append(avg_loss)
    #iter_losses = torch.stack(iter_losses)
    
modelPath = os.path.join(outDir, f'{d}_{v}', 'model.pth') 
torch.save(flow, modelPath)
print(f"Finished training for {n_iters} iterations")

#iter_losses = np.array(iter_losses.detach().cpu())
#iter_losses=np.array(iter_losses)

z_train=torch.cat(z_train, dim=0)
z_train=np.array(z_train.detach().cpu())

#print('Shape of avg losses:', iter_losses.shape)
print('Shape of z_train:', z_train.shape)

# Write results
np.save(outDir + "/z_train.npy", z_train)
#np.save(out_path + "/avg_losses.npy",losses)

print("Finished writing results.")
