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


class Muon(torch.optim.Optimizer):
    """Muon optimizer for matrix-shaped hidden weights.

    Use AdamW for embeddings, output heads, biases, and normalization weights.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, weight_decay: float = 0.0, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
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
                buf.mul_(momentum).add_(grad)
                update = zeropower_via_newtonschulz5(buf, steps=ns_steps)
                if p.ndim >= 2:
                    rows, cols = p.shape[0], p.numel() // p.shape[0]
                    update = update * math.sqrt(max(1.0, rows / cols))
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
