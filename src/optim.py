from __future__ import annotations

import math

import torch


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz orthogonalization used by Muon for 2D updates."""
    original_shape = g.shape
    x = g.float().reshape(g.shape[0], -1)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xx_t = x @ x.T
        x = a * x + (b * xx_t + c * (xx_t @ xx_t)) @ x
    if transposed:
        x = x.T
    return x.reshape(original_shape).type_as(g)


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float, ns_steps: int, nesterov: bool) -> torch.Tensor:
    momentum.lerp_(grad, 1.0 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    rows, cols = update.shape[0], update.numel() // update.shape[0]
    return update * math.sqrt(max(1.0, rows / cols))


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-shaped hidden weights.

    Use AdamW for embeddings, output heads, biases, and normalization weights.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
        nesterov: bool = True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            nesterov=nesterov,
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
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]
            nesterov = group["nesterov"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                if weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                grad = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                update = muon_update(grad, buf, beta=momentum, ns_steps=ns_steps, nesterov=nesterov)
                p.add_(update, alpha=-lr)
        return loss


def split_muon_params(model: torch.nn.Module):
    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embedding" not in name and "fc_out" not in name:
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params
