import torch
import torch.nn.functional as F
import pdb


def nce_loss(
        z1: torch.Tensor,
        z2: torch.Tensor,
        temperature: float = 1.0
        ) -> torch.Tensor:
    """
    PyTorch implementation of the NT-Xent loss introduced in 
    https://proceedings.mlr.press/v119/chen20j/chen20j.pdf

    Args:
        z1: embedding from view 1 (Tensor) of shape (bsz, dim).
        z2: embedding from view 2 (Tensor) of shape (bsz, dim).
        temperature: a floating number for temperature scaling.
    """
    LARGE_NUM = 1e9
    SMALL_NUM = 1e-9

    # TODO: implement NT-Xent loss

    loss = F.cross_entropy(sim, targets)
    
    return loss

    