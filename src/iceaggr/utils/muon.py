"""Muon optimizer — momentum SGD orthogonalized by Newton–Schulz.

Muon ("MomentUm Orthogonalized by Newton-schulz") updates each 2D weight matrix
by taking an SGD-momentum step and then replacing that update with its nearest
semi-orthogonal matrix, computed with a fixed number of Newton–Schulz iterations.
Orthogonalizing the update equalizes the per-direction step size, which trains
transformer hidden matrices faster than AdamW at matched cost.

The quintic Newton–Schulz iteration and its coefficients are taken from Keller
Jordan's reference implementation (https://github.com/KellerJordan/Muon, of
modded-nanogpt lineage; writeup at https://kellerjordan.github.io/posts/muon/).
As in the reference the iteration runs in bf16 — it is robust to the reduced
precision and this is where most of the compute lives.

Muon is meant for the 2D hidden weight matrices of a transformer *only*. Token
embeddings, the output head, gains/biases, and any parameter with <2 dims train
poorly under orthogonalization and belong in a standard optimizer (AdamW here),
which is why the training script pairs Muon with an AdamW side-group. See
``split_muon_params`` for the split this repo uses.
"""

from typing import Iterable

import torch
from torch import Tensor

# Quintic Newton–Schulz coefficients (Keller Jordan / modded-nanogpt). Tuned so
# the iteration pushes singular values toward 1 with a steep slope near 0.
_NS_ABC = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5) -> Tensor:
    """Orthogonalize ``G`` via a quintic Newton–Schulz iteration (run in bf16).

    Returns a matrix the same shape as ``G`` whose singular values are all close
    to 1 (i.e. approximately semi-orthogonal: orthonormal rows if wide, columns
    if tall). Batched leading dims are supported. The input is normalized by its
    Frobenius norm first so the spectral norm starts ``<= 1``, which is the
    region where this iteration converges.
    """
    assert G.ndim >= 2, "Newton–Schulz orthogonalization needs a matrix"
    a, b, c = _NS_ABC
    X = G.bfloat16()
    # Operate on the wide orientation so the Gram matrix X @ X^T is the smaller
    # of the two; transpose back at the end.
    transpose = X.size(-2) > X.size(-1)
    if transpose:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Momentum SGD with Newton–Schulz-orthogonalized updates, for 2D weights.

    Args:
        params: iterable of **2D** weight tensors (see the module docstring — do
            not pass embeddings / heads / scalars; use AdamW for those).
        lr: learning rate (nanochat/Muon default 0.02).
        momentum: SGD momentum coefficient (default 0.95).
        nesterov: use the Nesterov-style momentum of the reference (default True).
        ns_steps: number of Newton–Schulz iterations (default 5).
        weight_decay: decoupled weight decay (default 0.0).

    The per-parameter update is scaled by ``max(1, rows / cols) ** 0.5`` so
    non-square matrices get a step size comparable to square ones, matching the
    reference implementation.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
    ):
        defaults = dict(
            lr=lr, momentum=momentum, nesterov=nesterov,
            ns_steps=ns_steps, weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                assert g.ndim == 2, (
                    f"Muon expects 2D weight matrices, got shape {tuple(p.shape)}"
                )
                state = self.state[p]
                buf = state.get("momentum_buffer")
                if buf is None:
                    buf = state["momentum_buffer"] = torch.zeros_like(g)

                # EMA momentum, then (optional) Nesterov lookahead. lerp_(x, w)
                # computes (1 - w) * self + w * x in place.
                buf.lerp_(g, 1.0 - momentum)
                update = g.lerp_(buf, momentum) if nesterov else buf

                update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5

                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)
                p.add_(update.to(p.dtype), alpha=-lr * scale)

        return loss


def split_muon_params(module: torch.nn.Module):
    """Split a model's params into (muon, adamw) groups.

    Muon group: the 2D weight matrices inside ``module.blocks`` (attention
    ``c_q/c_k/c_v/c_proj`` and FFN ``c_fc/c_proj``). AdamW group: everything else
    — the CLS token, input projection, output/vMF head, per-layer scalars, any
    norm gains or biases, and every <2D or non-block parameter.

    Accepts a plain or ``torch.compile``-wrapped module (the ``_orig_mod`` prefix
    is stripped so names match either way). Returns ``(muon_params, adamw_params,
    muon_names)`` where ``muon_names`` is for logging.
    """
    base = getattr(module, "_orig_mod", module)
    muon_params, adamw_params, muon_names = [], [], []
    for name, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("blocks.") and p.ndim == 2:
            muon_params.append(p)
            muon_names.append(name)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params, muon_names


class MultiOptimizer:
    """Presents the ``torch.optim.Optimizer`` interface over several optimizers.

    Lets the training loop drive a Muon optimizer and an AdamW side-group as if
    they were one. Only the methods the loop uses are provided; ``param_groups``
    concatenates the groups so an LR scheduler / logging can see all of them.
    """

    def __init__(self, optimizers):
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        return [g for opt in self.optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def state_dict(self):
        return {"optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict):
        for opt, sd in zip(self.optimizers, state_dict["optimizers"]):
            opt.load_state_dict(sd)


class MultiScheduler:
    """Steps several LR schedulers in lockstep (one per underlying optimizer)."""

    def __init__(self, schedulers):
        self.schedulers = list(schedulers)

    def step(self, *args, **kwargs):
        for sch in self.schedulers:
            sch.step(*args, **kwargs)

    def get_last_lr(self):
        return [lr for sch in self.schedulers for lr in sch.get_last_lr()]

    def state_dict(self):
        return {"schedulers": [sch.state_dict() for sch in self.schedulers]}

    def load_state_dict(self, state_dict):
        for sch, sd in zip(self.schedulers, state_dict["schedulers"]):
            sch.load_state_dict(sd)
