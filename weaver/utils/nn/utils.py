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

#@torch.jit.script
def fgsm_attack(data: torch.Tensor,
                data_grad: torch.Tensor,
                eps_fgsm: float,
                eps_min: torch.Tensor,
                eps_max: torch.Tensor,
                mean: float = 1):

    maxd = eps_max;
    mind = eps_min;
    ## if there are infinite values, take max and min from data batch
    index_inf_min = (eps_min == float("Inf")).nonzero().squeeze();
    index_inf_max = (eps_max == float("Inf")).nonzero().squeeze();
    if index_inf_min.nelement() or index_inf_max.nelement():
        maxtmp, _ = torch.max(data,dim=2);
        mintmp, _ = torch.min(data,dim=2);
        maxtmp, _ = torch.max(maxtmp,dim=0);
        mintmp, _ = torch.max(mintmp,dim=0);
        maxd[index_inf_max] = maxtmp;
        mind[index_inf_min] = mintmp;
    ## build the final fgsm inputs
    maxd = maxd.unsqueeze(0).unsqueeze(2)
    mind = mind.unsqueeze(0).unsqueeze(2)
    maxd = torch.repeat_interleave(maxd,data.size(dim=0),dim=0);
    maxd = torch.repeat_interleave(maxd,data.size(dim=2),dim=2);
    mind = torch.repeat_interleave(mind,data.size(dim=0),dim=0);
    mind = torch.repeat_interleave(mind,data.size(dim=2),dim=2);
    output = data+data_grad*torch.normal(mean=mean,std=eps_fgsm,size=data.shape).to(data.device)*torch.full(data.shape,eps_fgsm).to(data.device)*(maxd-mind);
    return output
