from math import cos, pi, sqrt
from typing import IO, BinaryIO, Callable, Iterable, Optional
import torch
import torch.nn as nn
from numpy.typing import NDArray
import numpy as np
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim.optimizer import ParamsT
import os

MODEL_DICT = "MODEL_DICT"
OPTIM_DICT = "OPTIM_DICT"
ITERATION = "ITERATION"

def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[Tensor, ""]:
    stabilised = inputs - torch.amax(inputs, dim=-1, keepdim=True)
    chosens = torch.gather(stabilised, -1, targets.unsqueeze(-1))
    under = torch.logsumexp(stabilised, dim=-1, keepdim=True)
    batches = - (chosens - under)
    # thank https://jaykmody.com/blog/stable-softmax/ for this
    return torch.mean(batches)

class AdamW(torch.optim.Optimizer):
    def __init__(self, params: ParamsT, lr: float = 0.001, betas: tuple[float, float] = (0.9, 0.999), weight_decay: float = 0.01, eps=1e-8) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "reg": weight_decay, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            b1, b2 = group['betas']
            reg = group['reg']
            eps = group['eps']
            for param in group['params']:
                if param.grad is None:
                    continue
                # use state to store moment estimates m and v, along with time t
                state = self.state[param]
                t = state.get('t', 1)
                grad = param.grad.data
                a_t = lr * (sqrt(1-(b2**t))/(1-(b1**t)))
                
                param.data -= lr * reg * param.data
                state['m'] = b1 * state.get('m', torch.zeros_like(param)) + (1-b1)*grad
                state['v'] = b2 * state.get('v', torch.zeros_like(param)) + (1-b2)*(grad**2)
                param.data -= a_t * (state['m'])/(torch.sqrt(state['v']) + eps)
                state['t'] = t + 1
        return loss

def learning_rate_schedule(iteration: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if iteration < warmup_iters:
        # warm up
        return (iteration/warmup_iters)*max_lr
    elif warmup_iters <= iteration and iteration <= cosine_cycle_iters:
        # cosine
        return min_lr + 0.5 * (1 + cos((iteration - warmup_iters)/(cosine_cycle_iters-warmup_iters) * pi))*(max_lr-min_lr)
    else:
        return min_lr

def grad_clip(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    norm = 0.0
    for param in parameters:
        if param.grad is not None:
            norm += (param.grad**2).sum()
    norm = sqrt(norm)
    if norm < max_l2_norm:
        return
    clip_coef = max_l2_norm / (norm + eps)
    for param in parameters:
        if param.grad is not None:
            param.grad *= clip_coef

def get_batch(dataset: NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    indexes = np.random.randint(dataset.shape[0]-context_length, size=batch_size)
    x = torch.stack([
            torch.from_numpy(dataset[i : i + context_length].copy()) for i in indexes
        ])
    y = torch.stack([
            torch.from_numpy(dataset[i +1: i + 1 +context_length].copy()) for i in indexes
        ])
    if "cuda" in device:
        x.pin_memory().to(device)
        y.pin_memory().to(device)
    else:
        x.to(device)
        y.to(device)
    return x, y

def save_checkpoint(model: nn.Module, optimiser: torch.optim.Optimizer, iteration: int, out : str | os.PathLike | BinaryIO | IO[bytes]):
    model_dict = model.state_dict()
    optim_dict = optimiser.state_dict()
    output_obj = {MODEL_DICT: model_dict, OPTIM_DICT: optim_dict, ITERATION: iteration}
    torch.save(output_obj, out)

def load_checkpoint(model: nn.Module, optimiser: torch.optim.Optimizer, src : str | os.PathLike | BinaryIO | IO[bytes]) -> int:
    load_dict = torch.load(src)
    iteration = load_dict[ITERATION]
    model.load_state_dict(load_dict[MODEL_DICT])
    optimiser.load_state_dict(load_dict[OPTIM_DICT])
    return iteration

