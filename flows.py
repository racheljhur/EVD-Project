"""Continuous flows and transformations."""
import torch
import torch.nn as nn
from functools import partial
from math import pi
from torch import Tensor
from torch.distributions import Transform

from zuko.distributions import DiagNormal
from zuko.lazy import Flow, LazyTransform, UnconditionalDistribution
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast


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

class zukoMLP(nn.Sequential):
    def __init__(self, in_features, out_features,
                 hidden_features=(64, 64, 64), activation=nn.ELU, p_drop=0.0):
        layers, d = [], in_features
        for h in hidden_features:
            layers += [nn.Linear(d, h), activation()]
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))
            d = h
        layers.append(nn.Linear(d, out_features))
        super().__init__(*layers)


class FFJTransform(LazyTransform):
    r"""Creates a lazy free-form Jacobian (FFJ) transformation.

    See also:
        :class:`zuko.transforms.FreeFormJacobianTransform`

    References:
        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        freqs: The number of time embedding frequencies.
        atol: The absolute integration tolerance.
        rtol: The relative integration tolerance.
        exact: Whether the exact log-determinant of the Jacobian or an unbiased
            stochastic estimate thereof is calculated.
        kwargs: Keyword arguments passed to :class:`zuko.nn.MLP`.

    Example:
        >>> t = FFJTransform(3, 4)
        >>> t
        FFJTransform(
          (ode): MLP(
            (0): Linear(in_features=13, out_features=64, bias=True)
            (1): ELU(alpha=1.0)
            (2): Linear(in_features=64, out_features=64, bias=True)
            (3): ELU(alpha=1.0)
            (4): Linear(in_features=64, out_features=3, bias=True)
          )
        )
        >>> x = torch.randn(3)
        >>> x
        tensor([ 0.6365, -0.3181,  1.1519])
        >>> c = torch.randn(4)
        >>> y = t(c)(x)
        >>> t(c).inv(y)
        tensor([ 0.6364, -0.3181,  1.1519],
               grad_fn=<AdaptiveCheckpointAdjointBackward>)
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        freqs: int = 3,
        atol: float = 1e-4,
        rtol: float = 1e-3,
        exact: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        kwargs.setdefault("activation", nn.ELU)

        self.ode = zukoMLP(features + context + 2 * freqs, features, **kwargs)

        self.register_buffer("times", torch.tensor((0.0, 1.0)))
        self.register_buffer("freqs", torch.arange(1, freqs + 1) * pi)

        self.atol = atol
        self.rtol = rtol
        self.exact = exact

    def f(self, t: Tensor, x: Tensor, c: Tensor | None = None) -> Tensor:
        t = self.freqs * t[..., None]
        t = torch.cat((t.cos(), t.sin()), dim=-1)

        if c is None:
            x = torch.cat(broadcast(t, x, ignore=1), dim=-1)
        else:
            x = torch.cat(broadcast(t, x, c, ignore=1), dim=-1)

        return self.ode(x)

    def forward(self, c: Tensor | None = None) -> Transform:
        if self.training:
            phi = self.parameters() if c is None else (c, *self.parameters())
        else:
            phi = () if c is None else (c,)

        return FreeFormJacobianTransform(
            f=partial(self.f, c=c),
            t0=self.times[0],
            t1=self.times[1],
            phi=phi,
            atol=self.atol,
            rtol=self.rtol,
            exact=self.exact,
        )


class CNF(Flow):
    r"""Creates a continuous normalizing flow (CNF) with a free-form Jacobian
    transformation.

    References:
        | Neural Ordinary Differential Equations (Chen el al., 2018)
        | https://arxiv.org/abs/1806.07366

        | FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models (Grathwohl et al., 2018)
        | https://arxiv.org/abs/1810.01367

    Arguments:
        features: The number of features.
        context: The number of context features.
        kwargs: Keyword arguments passed to :class:`FFJTransform`.
    """

    def __init__(
        self,
        features: int,
        context: int = 0,
        **kwargs,
    ) -> None:
        transform = FFJTransform(
            features=features,
            context=context,
            **kwargs,
        )

        base = UnconditionalDistribution(
            DiagNormal,
            loc=torch.zeros(features),
            scale=torch.ones(features),
            buffer=True,
        )

        super().__init__(transform, base)

