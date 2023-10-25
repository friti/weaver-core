import numpy as np
import awkward as ak
import tqdm
import time
import torch
import gc
import ast

def _flatten_label(label, mask=None):
    if label.ndim > 1:
        label = label.view(-1)
        if mask is not None:
            label = label[mask.view(-1)]
    return label


def _flatten_preds(preds, mask=None, label_axis=1):
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes                                                                                                                                                             
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]
    return preds

@torch.jit.script
def fgsm_attack(data: torch.Tensor,
                data_grad: torch.Tensor,
                eps_fgsm: float,
                eps_min: torch.Tensor,
                eps_max: torch.Tensor,
                mean: float = 1):

    maxd = eps_max;
    mind = eps_min;
    maxd = maxd.unsqueeze(0).unsqueeze(2)
    mind = mind.unsqueeze(0).unsqueeze(2)
    maxd = torch.repeat_interleave(maxd,data.size(dim=0),dim=0);
    maxd = torch.repeat_interleave(maxd,data.size(dim=2),dim=2);
    mind = torch.repeat_interleave(mind,data.size(dim=0),dim=0);
    mind = torch.repeat_interleave(mind,data.size(dim=2),dim=2);
    output = data+data_grad*torch.normal(mean=mean,std=eps_fgsm,size=data.shape).to(data.device)*torch.full(data.shape,eps_fgsm).to(data.device)*(maxd-mind);
    return output


@torch.jit.script
def fngm_attack(data: torch.Tensor,
                data_grad: torch.Tensor,
                eps_fgsm: float,
                eps_min: torch.Tensor,
                eps_max: torch.Tensor,
                power: float = 2):

    maxd = eps_max;
    mind = eps_min;
    maxd = maxd.unsqueeze(0).unsqueeze(2)
    mind = mind.unsqueeze(0).unsqueeze(2)
    maxd = torch.repeat_interleave(maxd,data.size(dim=0),dim=0);
    maxd = torch.repeat_interleave(maxd,data.size(dim=2),dim=2);
    mind = torch.repeat_interleave(mind,data.size(dim=0),dim=0);
    mind = torch.repeat_interleave(mind,data.size(dim=2),dim=2);
    ## power is degree for normalization, take absolute val, collapse particle dimension
    data_grad = data_grad.nan_to_num();
    norm = data_grad.abs().pow(power).view(data_grad.size(0),-1).sum(dim=1).pow(1./power);
    ## avoid divisions per zero
    norm = torch.max(norm, torch.ones_like(norm) * 1e-12).view(-1,1,1);
    output = data+data_grad*(1./norm)*torch.full(data.shape,eps_fgsm).to(data.device)*(maxd-mind);
    return output
