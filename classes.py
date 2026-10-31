"""Useful functions
"""
import torch
import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

import numpy as np
from torch import Tensor
import zuko
import torch.nn as nn
from datetime import date
from data import *

import math
import warnings
from functools import partial
from typing import Optional, Union
import ot as pot
import argparse

# TODO: CHANGE THESE LATER!
# HARDCODING THE VALUES BAD
d = date.today()
ver = 0
dim = 1
cdim = 10

# Performance evaluation metrics
def compute_nmae(predictions, gr_truth, torch_tensor=True):
   '''compute nmae
      parameters:
        predictions-ml predictions
        gr_truth-training or testing dataset
        torch_tensor-boolean flag signaling if inputs are torch tensors or numpy ndarrays
   '''
   if torch_tensor:
        if predictions.size() != gr_truth.size():
            raise ValueError(f"Shape mismatch: predictions {predictions.size()} vs gr_truth {gr_truth.size()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def w2_dist(X, Y, numItermax=1_000_000):
    # Computes W_2 distance. Note ot.emd2 computes squared w2 distance.

    n = X.shape[0]
    a, b = np.ones((n,)) / n, np.ones((n,)) / n

    M = pot.dist(X, Y)
    W = pot.emd2(a, b, M, numItermax=numItermax)

    return np.sqrt(W)

def sample_moons(n, theta):
    theta = np.radians(theta)
    rot_mat = torch.tensor([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta),  np.cos(theta)]],dtype=torch.float32)
    
    x0, y0 = generate_moons(n, noise=0.2)
    x0 = 3*(x0 - 0.5)
    x0 = torch.matmul(x0, rot_mat)
    return x0, y0


def sample_pcs(n, timestep):
    """Params:
            n: batch size
        Returns
            Mini batches of (x0,y0) pairs

        Note that this function assumes that data is imported as a module,
        within which pc_sc_0, ym_0, pc_sc_1, ym_1 are defined with these
        exact variable names.
    """
    if timestep=="t0":
        x0,y0=pc_sc_0,ym_0
        x0=torch.tensor(x0, dtype=torch.float32).to(device)
        y0=torch.tensor(y0, dtype=torch.float32).to(device)

        dataset=TensorDataset(x0, y0)
        loader=DataLoader(dataset, batch_size=n, shuffle=True)
        x0,y0=next(iter(loader))
    else:
        x1,y1=pc_sc_1,ym_1
        x1=torch.tensor(x1, dtype=torch.float32).to(device)
        y1=torch.tensor(y1, dtype=torch.float32).to(device)

        dataset=TensorDataset(x1, y1)
        loader=DataLoader(dataset, batch_size=n, shuffle=True)
        x0,y0=next(iter(loader)) # This is just a placeholder so I can use the same function for both t0 and t1 cases.

    return x0,y0


class SDE_uncond(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, model, sigma=1.0, sigma_fac=1.0):
        super().__init__()
        self.model = model
        self.sigma = sigma
        self.sigma_fac = sigma_fac
        #self.cond = cond

    # Drift
    def f(self, t, y):
        t = t.repeat(y.shape[0])[:, None]
        vt, st = self.model(y,t)
        return vt + st

    # Diffusion
    def g(self, t, y):
        out = torch.ones_like(y) * self.sigma
        out[:,0] = out[:,0]*self.sigma_fac
        return out

class MLP(torch.nn.Module):
    def __init__(self, dim, cdim, w=64, p_drop = 0.1):
        super().__init__()
        self.dim = dim
        self.cdim = cdim
        self.model_type = 'otcfm'
        
        # note that, the extreme mp stresses
        # have positive spearman rank correlations
        # with many pcs (from the bigData pca model).
        # this is why i've kept this fixed to ReLu.
        # for monotonically negative relationships, 
        # you would just flip the sign of inputs.
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + cdim + 1, w),
            torch.nn.ReLU(), nn.Dropout(p_drop),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(), nn.Dropout(p_drop),
            torch.nn.Linear(w, w),
            torch.nn.ReLU(), nn.Dropout(p_drop),
            torch.nn.Linear(w, dim),
        )

    def forward(self, x, y, t):
        in_x = torch.cat([x, y, t], dim=-1)
        out = self.net(in_x)
        return out

class MLP_uncond(torch.nn.Module):
    def __init__(self, dim, w=256):
        super().__init__()
        self.dim = dim
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + 1, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, w),
            torch.nn.GELU(),
            torch.nn.Linear(w, dim),
        )

    def forward(self,x, t):
        in_x = torch.cat([x, t], dim=-1)
        out = self.net(in_x)
        return out

class MLP_attn(torch.nn.Module):
    def __init__(self, dim, cdim, edim, layers=3, w=64, w_embed=32, num_heads=8):
        super(MLP, self).__init__()

        self.first_layer = torch.nn.Sequential(
            torch.nn.Linear(edim + dim, w),
            torch.nn.LayerNorm(w),
            torch.nn.GELU(),
        )

        self.cond_emb_y = torch.nn.Sequential(
            torch.nn.Linear(cdim, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )

        self.cond_emb_t = torch.nn.Sequential(
            torch.nn.Linear(1, edim),
            torch.nn.LayerNorm(edim),
            torch.nn.GELU(),
        )

        self.attn = torch.nn.MultiheadAttention(embed_dim=edim, num_heads=num_heads, batch_first=True)
        self.ln = torch.nn.LayerNorm(edim)

        self.blocks = torch.nn.ModuleList()
        for i in range(layers):
            self.blocks.append(ResidualBlock(w + edim, w))

        self.last_layer = torch.nn.Linear(w + edim, dim)


    def forward(self, x, y, t):
        y = self.cond_emb_y(y)
        t = self.cond_emb_t(t)
        y_residual = y
        t_residual = t

        y = self.ln(y)
        y, _ = self.attn(y, y, y)      # self-attention conditioning
        y += y_residual
        y_residual = y

        y = self.ln(y)
        t = self.ln(t)
        yt, _ = self.attn(y, t, t)      # cross-attention conditoning / time
        yt += y_residual + t_residual

        in_x = torch.cat([x, yt], -1)
        out = self.first_layer(in_x)

        for i in range(len(self.blocks)):
            out = self.blocks[i](out, yt)

        in_x = torch.cat([out, yt], -1)
        out = self.last_layer(in_x)

        return out

class torchdyn_wrapper(torch.nn.Module):
    def __init__(self, model, y):
        super().__init__()
        self.model = model
        self.y = y
        
    def forward(self, t, x, args=None):  
        return self.model(x, self.y, t.repeat(x.shape[0])[:, None]) 

def plot_trajectories(traj, y):
    n = 2000
    y = y[:n].astype('int')
    
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    colors = cmap(norm(y)) 
    
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, 0], traj[0, :n, 1], s=10, alpha=0.8, c="black",zorder=5)
    plt.scatter(traj[-1, :n, 0], traj[-1, :n, 1], s=10, alpha=1, c="maroon",zorder=10)
    for i in range(n):
        plt.plot(traj[:, i, 0], traj[:, i, 1], color=colors[i], alpha=0.1,zorder=0)
    
    plt.legend([r"$p_0$", r"$p_1$", r"$x_{t} \vert x_{0}$"])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.show()

def plot_trajectories_pcs(traj, j, y, filename):
    n = int(1125/4)
    y = y[:n].astype('int')
    
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=y.min(), vmax=y.max())
    colors = cmap(norm(y)) 
    
    plt.figure(figsize=(6, 6))
    plt.scatter(traj[0, :n, j], traj[0, :n, j+1], s=10, alpha=0.8, c="black",zorder=5)
    plt.scatter(traj[-1, :n, j], traj[-1, :n, j+1], s=10, alpha=1, c="maroon",zorder=10)
    for i in range(n):
        plt.plot(traj[:, i, j], traj[:, i, j+1], color=colors[i], alpha=0.1,zorder=0)
    
    plt.legend([r"$p_0$", r"$p_1$", r"$x_{t} \vert x_{0}$"])
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(filename)

def plot_couplings(x0,x1,y0,y1,filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:,0].detach().cpu().numpy(),x0[:,1].detach().cpu().numpy(), c="black")
    plt.scatter(x1[:,0].detach().cpu().numpy(),x1[:,1].detach().cpu().numpy(), c="maroon")
    plt.plot([x0[:,0].detach().cpu().numpy(),x1[:,0].detach().cpu().numpy()],[x0[:,1].detach().cpu().numpy(),x1[:,1].detach().cpu().numpy()], color="black", alpha=0.2)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(filename)

def plot_couplings(x0,x1,filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:,0].detach().cpu().numpy(),x0[:,1].detach().cpu().numpy(), c="black")
    plt.scatter(x1[:,0].detach().cpu().numpy(),x1[:,1].detach().cpu().numpy(), c="maroon")
    plt.plot([x0[:,0].detach().cpu().numpy(),x1[:,0].detach().cpu().numpy()],[x0[:,1].detach().cpu().numpy(),x1[:,1].detach().cpu().numpy()], color="black", alpha=0.2)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(filename)

def plot_couplings_pcs(x0,x1,i,j,title,filename):
    plt.figure(figsize=(6, 6))
    plt.scatter(x0[:,i].detach().cpu().numpy(),x0[:,j].detach().cpu().numpy(), c="black", label="x0")
    plt.scatter(x1[:,i].detach().cpu().numpy(),x1[:,j].detach().cpu().numpy(), c="maroon", label="x1")
    plt.plot([x0[:,i].detach().cpu().numpy(),x1[:,i].detach().cpu().numpy()],[x0[:,j].detach().cpu().numpy(),x1[:,j].detach().cpu().numpy()], color="black", alpha=0.2)
    # plt.xticks([])
    # plt.yticks([])
    if i==3:
        plt.xlim(-2,3)
        plt.ylim(-1.5,1.5)
    plt.xlabel(rf"$\alpha_{i+1}$")    
    plt.ylabel(rf"$\alpha_{j+1}$")  
    plt.title(f"{title}")
    plt.legend()
    plt.savefig(filename)


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        num_threads: int or str, optional
            number of threads to use for the "exact" OT solver. If "max", uses
            the maximum number of threads.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.sinkhorn, reg=reg)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_knopp_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0, x1) ** 2
        if self.normalize_cost:
            M = M / M.max()  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M.detach().cpu().numpy())
        if not np.all(np.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if np.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = np.ones_like(p) / p.size
        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = np.random.choice(
            pi.shape[0] * pi.shape[1], p=p, size=batch_size, replace=replace
        )
        return np.divmod(choices, pi.shape[1])

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_scipy(self, x0, x1):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch using scipy and draw source and target samples from pi $(x,z) \sim \pi$.

        This sampler has two advantages:
        * Reduced variance compared to sampling from the OT plan
        * Preserves the order of x1 by construction
        * Preserves entire batch if x0 and x1 have the same size

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        import scipy

        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = torch.cdist(x0.detach(), x1.detach()) ** 2
        if self.normalize_cost:
            M = M / M.max()
        _, j = scipy.optimize.linear_sum_assignment(M.cpu().numpy())
        pi_x0 = x0
        pi_x1 = x1[j]
        return pi_x0, pi_x1

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs)
            represents the source label minibatch
        y1 : Tensor, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [np.arange(X.shape[0])]
        for pi in pis:
            j = []
            for i in indices[-1]:
                j.append(np.random.choice(pi.shape[1], p=pi[i] / pi[i].sum()))
            indices.append(np.array(j))

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = np.stack(to_return, axis=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    x0 = torch.as_tensor(x0, dtype=torch.float32)
    x1 = torch.as_tensor(x1, dtype=torch.float32)
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    ret = ot_fn(a, b, M.detach().cpu().numpy(), numItermax=int(1e7))
    if power == 2:
        ret = math.sqrt(ret)
    return ret


class ExactGPModel(gpytorch.models.ExactGP):
  def __init__(self, train_x, train_y, likelihood):
      super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
      self.mean_module=gpytorch.means.ConstantMean()
      #self.covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=cdim))
      #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.LinearKernel(ard_num_dims=cdim))
      self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures = 3, ard_num_dims = cdim)


  def forward(self, x):
      mean_x = self.mean_module(x)
      covar_x = self.covar_module(x)
      return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class SparseGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SparseGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module=gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=cdim))

        #self.base_covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures = 2, ard_num_dims = cdim)
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:500, :].clone(), likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# cdim is arbitrarily defined to instantiate these constructors.
# it gets overwritten in the actual train and evaluation files.
class VariationalGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VariationalGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = cdim))
        #self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures = 3, ard_num_dims = cdim)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)







