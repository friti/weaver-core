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


def _flatten_preds(model_output, label=None, mask=None, label_axis=1):
    if not isinstance(model_output, tuple):
        # `label` and `mask` are provided as function arguments
        preds = model_output
    else:
        if len(model_output == 2):
            # use `mask` from model_output instead
            # `label` still provided as function argument
            preds, mask = model_output
        elif len(model_output == 3):
            # use `label` and `mask` from model output
            preds, label, mask = model_output
    if preds.ndim > 2:
        # assuming axis=1 corresponds to the classes                                                                                                                                             
        preds = preds.transpose(label_axis, -1).contiguous()
        preds = preds.view((-1, preds.shape[-1]))
        if mask is not None:
            preds = preds[mask.view(-1)]

    if label is not None:
        label = _flatten_label(label, mask)
         
    return preds, label, mask

@torch.jit.script
def fgsm_attack(data: torch.Tensor,
                data_grad: torch.Tensor,
                eps_fgsm: float,
                eps_min: torch.Tensor,
                eps_max: torch.Tensor,
                mean: float = 1):

    maxd = eps_max.unsqueeze(0).unsqueeze(2)
    mind = eps_min.unsqueeze(0).unsqueeze(2)
    maxd = torch.repeat_interleave(maxd,data.size(dim=0),dim=0);
    maxd = torch.repeat_interleave(maxd,data.size(dim=2),dim=2);
    mind = torch.repeat_interleave(mind,data.size(dim=0),dim=0);
    mind = torch.repeat_interleave(mind,data.size(dim=2),dim=2);
    data_grad = data.grad.data.sign().detach();
    output = data+data_grad*torch.normal(mean=mean,std=eps_fgsm,size=data.shape).to(data.device,non_blocking=True)*torch.full(data.shape,eps_fgsm).to(data.device,non_blocking=True)*(maxd-mind);
    return output;
    

@torch.jit.script
def fngm_attack(data: torch.Tensor,
                data_grad: torch.Tensor,
                eps_fgsm: float,
                eps_min: torch.Tensor,
                eps_max: torch.Tensor,
                power: float = 2):

    maxd = eps_max.unsqueeze(0).unsqueeze(2)
    mind = eps_min.unsqueeze(0).unsqueeze(2)
    maxd = torch.repeat_interleave(maxd,data.size(dim=0),dim=0);
    maxd = torch.repeat_interleave(maxd,data.size(dim=2),dim=2);
    mind = torch.repeat_interleave(mind,data.size(dim=0),dim=0);
    mind = torch.repeat_interleave(mind,data.size(dim=2),dim=2);
    data_grad = data.grad.data.detach().nan_to_num();
    norm = data_grad.abs().pow(power).view(data_grad.size(0),-1).sum(dim=1).pow(1./power);
    norm = torch.max(norm, torch.ones_like(norm) * 1e-12).view(-1,1,1);
    output = data+data_grad*(1./norm)*torch.full(data.shape,eps_fgsm).to(data.device,non_blocking=True)*(maxd-mind);
    return output;
