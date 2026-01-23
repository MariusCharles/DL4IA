import torch
import torch.nn.functional as F
from torch.utils.flop_counter import FlopCounterMode

import numpy as np

from typing import Union, Tuple, Optional, List
from datetime import datetime

from torch.utils.data import DataLoader


def dates2doys(dates: list[str]):
    """
    Convert a list of dates (YYYY-MM-DD) to day of year (DOY).
    """
    doys = []
    for date in dates:
        month = date[5:7]
        day = date[8:]
        day = int(day) + 30*int(month)
        doys.append(day)
    return torch.tensor(doys).long()



def pad_tensor(x: torch.Tensor, l: int, pad_value=0.):
    ''' Adds padding to a tensor.
    '''
    padlen = l - x.shape[0]
    pad = [0 for _ in range(2 * len(x.shape[1:]))] + [0, padlen]
    return F.pad(x, pad=pad, value=pad_value)


def fill_ts(ts: torch.Tensor, doys: torch.Tensor, full_doys: torch.Tensor):
    ''' Fill the gaps in a time series with NaN values.
    Args:
        ts: time series with missing data
        doys: days of year of the time series
        full_doys: complete list of days of year (including missing dates)
    '''
    full_length = len(full_doys)
    ts = pad_tensor(ts, full_length, pad_value=torch.nan)
    missing_doys = torch.tensor(list(
        set(full_doys.tolist()) - set(doys.tolist())
    ))
    missing_doys, _ = missing_doys.sort()
    doys = torch.cat((doys, missing_doys))
    doys, indices = doys.sort()
    indices = indices.view(-1, 1, 1).repeat(1, ts.shape[1], ts.shape[2])
    ts = torch.gather(ts, index=indices, dim=0)
    return ts


def get_params(model: torch.nn.Module):
    '''TODO: compute the number of trainable parameters of a model.
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_flops(model, inp: Union[torch.Tensor, Tuple], with_backward=False):
    '''Credit: https://alessiodevoto.github.io/Compute-Flops-with-Pytorch-built-in-flops-counter/
    '''
    istrain = model.training
    model.eval()
    
    inp = inp if isinstance(inp, torch.Tensor) else torch.randn(inp)

    flop_counter = FlopCounterMode(mods=model, display=False, depth=None)
    with flop_counter:
        if with_backward:
            model(inp).sum().backward()
        else:
            model(inp)
    total_flops =  flop_counter.get_total_flops()
    if istrain:
        model.train()
    return total_flops

def rgb_render(
    data: np.ndarray,
    clip: int = 2,
    bands: Optional[List[int]] = None,
    norm: bool = True,
    dmin: Optional[np.ndarray] = None,
    dmax: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepare data for visualization with matplot lib, taken (with minor modifications) from the sensorsio repo by Julien Michel
    Source: https://framagit.org/jmichel-otb/sensorsio/-/blob/master/src/sensorsio/utils.py?ref_type=heads
    License: Apache License, Version 2.0

    :param data: nd_array of shape [bands, w, h]
    :param clip: clip percentile (between 0 and 100). Ignored if norm is False
    :bands: List of bands to extract (len is 1 or 3 for RGB)
    :norm: If true, clip a percentile at each end

    :returns: a tuple of data ready for matplotlib, dmin, dmax
    """
    if bands is None:
        bands = [2, 1, 0]
    assert len(bands) == 1 or len(bands) == 3
    assert 0 <= clip <= 100

    # Extract bands from data
    data_ready = np.take(data, bands, axis=0)
    out_dmin = None
    out_dmax = None
    # If normalization is on
    if norm:
        # Rescale and clip data according to percentile
        if dmin is None:
            out_dmin = np.percentile(data_ready, clip, axis=(1, 2))
        else:
            out_dmin = dmin
        if dmax is None:
            out_dmax = np.percentile(data_ready, 100 - clip, axis=(1, 2))
        else:
            out_dmax = dmax
        data_ready = np.clip((data_ready.transpose(1, 2, 0) - out_dmin) / (out_dmax - out_dmin), 0, 1)

    else:
        data_ready.transpose(1, 2, 0)

    # Strip of one dimension if number of bands is 1
    if data_ready.shape[-1] == 1:
        data_ready = data_ready[:, :, 0]

    return data_ready, out_dmin, out_dmax


def mean_attention(encoder: torch.nn.Module,
                   dataset: torch.utils.data.Dataset,
                   classe: int,
                   batch_size=32,
                   device="cuda"):
    """
    Compute the mean attention for one classe based on a dataset

    return:
        weight: numpy array, weight for which head 
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    encoder.eval()

    attn_sum = None
    n_samples = 0

    with torch.no_grad():
        for data, doys, labels in loader:
            mask = labels == classe
            if mask.sum() == 0:
                continue

            data = data[mask].to(device)
            doys = doys[mask].to(device)

            _, attn = encoder(data, doys)
            attn = attn.squeeze(2)  

            if attn_sum is None:
                attn_sum = attn.sum(dim=0)
            else:
                attn_sum += attn.sum(dim=0)

            n_samples += attn.size(0)

    return (attn_sum / n_samples).cpu().numpy()