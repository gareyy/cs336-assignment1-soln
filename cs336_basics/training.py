from math import sqrt
from typing import Callable, Optional
import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from torch.optim.optimizer import ParamsT

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
