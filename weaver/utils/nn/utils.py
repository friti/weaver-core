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
                mean: float = 1):
    maxd, _ = torch.max(data,dim=2);
    mind, _ = torch.min(data,dim=2);
    maxd, _ = torch.max(maxd,dim=0);
    mind, _ = torch.max(mind,dim=0);
    maxd = maxd.unsqueeze(0).unsqueeze(2)
    mind = mind.unsqueeze(0).unsqueeze(2)
    maxd = torch.repeat_interleave(maxd,data.size(dim=0),dim=0);
    maxd = torch.repeat_interleave(maxd,data.size(dim=2),dim=2);
    mind = torch.repeat_interleave(mind,data.size(dim=0),dim=0);
    mind = torch.repeat_interleave(mind,data.size(dim=2),dim=2);
    output = data+data_grad*torch.normal(mean=mean,std=eps_fgsm,size=data.shape)*torch.full(data.shape,eps_fgsm)*(maxd-mind);
    return output
