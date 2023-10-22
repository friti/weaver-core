''' Particle Transformer (ParT)
Paper: "Particle Transformer for Jet Tagging" - https://arxiv.org/abs/2202.03772
1;95;0c'''
import math
import random
import warnings
import copy
import torch
import torch.nn as nn
from functools import partial


@torch.jit.script
def delta_phi(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


@torch.jit.script
def delta_r2(eta1, phi1, eta2, phi2):
    return (eta1 - eta2)**2 + delta_phi(phi1, phi2)**2


def to_pt2(x, eps=1e-8):
    pt2 = x[:, :2].square().sum(dim=1, keepdim=True)
    if eps is not None:
        pt2 = pt2.clamp(min=eps)
    return pt2


def to_m2(x, eps=1e-8):
    m2 = x[:, 3:4].square() - x[:, :3].square().sum(dim=1, keepdim=True)
    if eps is not None:
        m2 = m2.clamp(min=eps)
    return m2


def atan2(y, x):
    sx = torch.sign(x)
    sy = torch.sign(y)
    pi_part = (sy + sx * (sy ** 2 - 1)) * (sx - 1) * (-math.pi / 2)
    atan_part = torch.arctan(y / (x + (1 - sx ** 2))) * sx ** 2
    return atan_part + pi_part


def to_ptrapphim(x, return_mass=True, eps=1e-8, for_onnx=False):
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    px, py, pz, energy = x.split((1, 1, 1, 1), dim=1)
    pt = torch.sqrt(to_pt2(x, eps=eps))
    phi = (atan2 if for_onnx else torch.atan2)(py, px)
    rapidity = 0.5 * torch.log(((energy+pz) / (energy-pz)).clamp(min=1e-20))
    if not return_mass:
        return torch.cat((pt, rapidity, phi), dim=1)
    else:
        m = torch.sqrt(to_m2(x, eps=eps))
        return torch.cat((pt, rapidity, phi, m), dim=1)


def boost(x, boostp4, eps=1e-8):
    # boost x to the rest frame of boostp4
    # x: (N, 4, ...), dim1 : (px, py, pz, E)
    p3 = -boostp4[:, :3] / boostp4[:, 3:].clamp(min=eps)
    b2 = p3.square().sum(dim=1, keepdim=True)
    gamma = (1 - b2).clamp(min=eps)**(-0.5)
    gamma2 = (gamma - 1) / b2
    gamma2.masked_fill_(b2 == 0, 0)
    bp = (x[:, :3] * p3).sum(dim=1, keepdim=True)
    v = x[:, :3] + gamma2 * bp * p3 + x[:, 3:] * gamma * p3
    return v


def p3_norm(p, eps=1e-8):
    return p[:, :3] / p[:, :3].norm(dim=1, keepdim=True).clamp(min=eps)


def pairwise_lv_fts(xi, xj, num_outputs=4, eps=1e-8, for_onnx=False):
    pti, rapi, phii = to_ptrapphim(xi, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)
    ptj, rapj, phij = to_ptrapphim(xj, False, eps=None, for_onnx=for_onnx).split((1, 1, 1), dim=1)

    delta = delta_r2(rapi, phii, rapj, phij).sqrt()
    lndelta = torch.log(delta.clamp(min=eps))
    if num_outputs == 1:
        return lndelta

    if num_outputs > 1:
        ptmin = ((pti <= ptj) * pti + (pti > ptj) * ptj) if for_onnx else torch.minimum(pti, ptj)
        lnkt = torch.log((ptmin * delta).clamp(min=eps))
        lnz = torch.log((ptmin / (pti + ptj).clamp(min=eps)).clamp(min=eps))
        outputs = [lnkt, lnz, lndelta]
    if num_outputs > 3:
        xij = xi + xj
        lnm2 = torch.log(to_m2(xij, eps=eps))
        outputs.append(lnm2)

    if num_outputs > 4:
        lnds2 = torch.log(torch.clamp(-to_m2(xi - xj, eps=None), min=eps))
        outputs.append(lnds2)

    # the following features are not symmetric for (i, j)
    if num_outputs > 5:
        xj_boost = boost(xj, xij)
        costheta = (p3_norm(xj_boost, eps=eps) * p3_norm(xij, eps=eps)).sum(dim=1, keepdim=True)
        outputs.append(costheta)

    if num_outputs > 6:
        deltarap = rapi - rapj
        deltaphi = delta_phi(phii, phij)
        outputs += [deltarap, deltaphi]

    assert (len(outputs) == num_outputs)
    return torch.cat(outputs, dim=1)

def build_sparse_tensor(uu, idx, seq_len):
    # inputs: uu (N, C, num_pairs), idx (N, 2, num_pairs)
    # return: (N, C, seq_len, seq_len)
    batch_size, num_fts, num_pairs = uu.size()
    idx = torch.min(idx, torch.ones_like(idx) * seq_len)
    i = torch.cat((
        torch.arange(0, batch_size, device=uu.device).repeat_interleave(num_fts * num_pairs).unsqueeze(0),
        torch.arange(0, num_fts, device=uu.device).repeat_interleave(num_pairs).repeat(batch_size).unsqueeze(0),
        idx[:, :1, :].expand_as(uu).flatten().unsqueeze(0),
        idx[:, 1:, :].expand_as(uu).flatten().unsqueeze(0),
    ), dim=0)
    return torch.sparse_coo_tensor(
        i, uu.flatten(),
        size=(batch_size, num_fts, seq_len + 1, seq_len + 1),
        device=uu.device).to_dense()[:, :, :seq_len, :seq_len]

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # From https://github.com/rwightman/pytorch-image-models/blob/18ec173f95aa220af753358bf860b16b6691edb2/timm/layers/weight_init.py#L8
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


class SequenceTrimmer(nn.Module):

    def __init__(self, enabled=False, target=(0.9, 1.02), **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        self.target = target
        self._counter = 0

    def forward(self, x, v=None, mask=None, uu=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # uu: (N, C', P, P)
        if mask is None:
            mask = torch.ones_like(x[:, :1])
        mask = mask.bool()

        if self.enabled:
            if self._counter < 5:
                self._counter += 1
            else:
                if self.training:
                    q = torch.min(torch.ones(1, device=mask.device), torch.rand(1, device=mask.device) * (self.target[1] - self.target[0]) + self.target[0])[0]
                    maxlen = torch.quantile(mask.type_as(x).sum(dim=-1), q).long()
                    rand = torch.rand_like(mask.type_as(x))
                    rand.masked_fill_(~mask, -1)
                    perm = rand.argsort(dim=-1, descending=True)  # (N, 1, P)
                    mask = torch.gather(mask, -1, perm)
                    x = torch.gather(x, -1, perm.expand_as(x))
                    if v is not None:
                        v = torch.gather(v, -1, perm.expand_as(v))
                    if uu is not None:
                        uu = torch.gather(uu, -2, perm.unsqueeze(-1).expand_as(uu))
                        uu = torch.gather(uu, -1, perm.unsqueeze(-2).expand_as(uu))
                else:
                    maxlen = mask.sum(dim=-1).max()
                maxlen = max(maxlen, 1)
                if maxlen < mask.size(-1):
                    mask = mask[:, :, :maxlen]
                    x = x[:, :, :maxlen]
                    if v is not None:
                        v = v[:, :, :maxlen]
                    if uu is not None:
                        uu = uu[:, :, :maxlen, :maxlen]

        return x, v, mask, uu


class Embed(nn.Module):
    def __init__(self, input_dim, dims, normalize_input=True, activation='gelu'):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim) if normalize_input else None
        module_list = []
        for dim in dims:
            module_list.extend([
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, dim),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
            ])
            input_dim = dim
        self.embed = nn.Sequential(*module_list)

    def forward(self, x):
        if self.input_bn is not None:
            # x: (batch, embed_dim, seq_len)
            x = self.input_bn(x)
            x = x.permute(2, 0, 1).contiguous()
        return self.embed(x)


class PairEmbed(nn.Module):
    def __init__(
            self, pairwise_lv_dim, pairwise_input_dim, dims,
            remove_self_pair=False, use_pre_activation_pair=True, mode='sum',
            normalize_input=True, activation='gelu', eps=1e-8,
            for_onnx=False):
        super().__init__()

        self.pairwise_lv_dim = pairwise_lv_dim
        self.pairwise_input_dim = pairwise_input_dim
        self.is_symmetric = (pairwise_lv_dim <= 5) and (pairwise_input_dim == 0)
        self.remove_self_pair = remove_self_pair
        self.mode = mode
        self.for_onnx = for_onnx
        self.pairwise_lv_fts = partial(pairwise_lv_fts, num_outputs=pairwise_lv_dim, eps=eps, for_onnx=for_onnx)
        self.out_dim = dims[-1]
        self.save_grad_inputs = False;

        if self.mode == 'concat':
            input_dim = pairwise_lv_dim + pairwise_input_dim
            module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
            for dim in dims:
                module_list.extend([
                    nn.Conv1d(input_dim, dim, 1),
                    nn.BatchNorm1d(dim),
                    nn.GELU(s) if activation == 'gelu' else nn.ReLU(),
                ])
                input_dim = dim
            if use_pre_activation_pair:
                module_list = module_list[:-1]
            self.embed = nn.Sequential(*module_list)
        elif self.mode == 'sum':
            if pairwise_lv_dim > 0:
                input_dim = pairwise_lv_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.embed = nn.Sequential(*module_list)

            if pairwise_input_dim > 0:
                input_dim = pairwise_input_dim
                module_list = [nn.BatchNorm1d(input_dim)] if normalize_input else []
                for dim in dims:
                    module_list.extend([
                        nn.Conv1d(input_dim, dim, 1),
                        nn.BatchNorm1d(dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    ])
                    input_dim = dim
                if use_pre_activation_pair:
                    module_list = module_list[:-1]
                self.fts_embed = nn.Sequential(*module_list)
        else:
            raise RuntimeError('`mode` can only be `sum` or `concat`')

    def forward(self, x, uu=None):
        # x: (batch, v_dim, seq_len)
        # uu: (batch, v_dim, seq_len, seq_len)
        assert (x is not None or uu is not None)
        with torch.set_grad_enabled(self.save_grad_inputs):
            if x is not None:
                batch_size, _, seq_len = x.size()
            else:
                batch_size, _, seq_len, _ = uu.size()
            if self.is_symmetric and not self.for_onnx:
                i, j = torch.tril_indices(seq_len, seq_len, offset=-1 if self.remove_self_pair else 0,
                                          device=(x if x is not None else uu).device)
                if x is not None:
                    x = x.unsqueeze(-1).repeat(1, 1, 1, seq_len)
                    xi = x[:, :, i, j]  # (batch, dim, seq_len*(seq_len+1)/2)
                    xj = x[:, :, j, i]                    
                    x = self.pairwise_lv_fts(xi, xj)
                if uu is not None:
                    # (batch, dim, seq_len*(seq_len+1)/2)
                    uu = uu[:, :, i, j]
            else:
                if x is not None:
                    x = self.pairwise_lv_fts(x.unsqueeze(-1), x.unsqueeze(-2))
                    if self.remove_self_pair:
                        i = torch.arange(0, seq_len, device=x.device)
                        x[:, :, i, i] = 0
                    x = x.view(-1, self.pairwise_lv_dim, seq_len * seq_len)
                if uu is not None:
                    uu = uu.view(-1, self.pairwise_input_dim, seq_len * seq_len)

            if self.mode == 'concat':
                if x is None:
                    pair_fts = uu
                elif uu is None:
                    pair_fts = x
                else:
                    pair_fts = torch.cat((x, uu), dim=1)

        if self.mode == 'concat':
            elements = self.embed(pair_fts)  # (batch, embed_dim, num_elements)
        elif self.mode == 'sum':
            if x is None:
                elements = self.fts_embed(uu)
            elif uu is None:
                elements = self.embed(x)
            else:
                elements = self.embed(x) + self.fts_embed(uu)

        if self.is_symmetric and not self.for_onnx:
            y = torch.zeros(batch_size, self.out_dim, seq_len, seq_len, dtype=elements.dtype, device=elements.device)
            y[:, :, i, j] = elements
            y[:, :, j, i] = elements
        else:
            y = elements.view(-1, self.out_dim, seq_len, seq_len)
        return y


class Block(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ffn_ratio=4,
                 dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                 add_bias_kv=False, activation='gelu',
                 scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.ffn_dim = embed_dim * ffn_ratio
        
        self.pre_attn_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            add_bias_kv=add_bias_kv,
        )
        self.post_attn_norm = nn.LayerNorm(embed_dim) if scale_attn else None
        self.dropout = nn.Dropout(dropout)

        self.pre_fc_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, self.ffn_dim)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.act_dropout = nn.Dropout(activation_dropout)
        self.post_fc_norm = nn.LayerNorm(self.ffn_dim) if scale_fc else None
        self.fc2 = nn.Linear(self.ffn_dim, embed_dim)

        self.c_attn = nn.Parameter(torch.ones(num_heads), requires_grad=True) if scale_heads else None
        self.w_resid = nn.Parameter(torch.ones(embed_dim), requires_grad=True) if scale_resids else None

    def forward(self, x, x_cls=None, padding_mask=None, attn_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            x_cls (Tensor, optional): class token input to the layer of shape `(1, batch, embed_dim)`
            padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, seq_len)` where padding
                elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        padding_mask = torch.zeros_like(padding_mask, dtype=x.dtype).masked_fill(padding_mask, float('-inf'))                
        if x_cls is not None:
            with torch.no_grad():
                padding_mask = torch.cat((torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1)
            # class attention: https://arxiv.org/pdf/2103.17239.pdf
            residual = x_cls
            u = torch.cat((x_cls, x), dim=0)  # (seq_len+1, batch, embed_dim)
            u = self.pre_attn_norm(u)
            x = self.attn(x_cls, u, u, key_padding_mask=padding_mask, need_weights=False)[0]  # (1, batch, embed_dim)
        else:
            residual = x
            x = self.pre_attn_norm(x)
            x = self.attn(x, x, x, key_padding_mask=padding_mask,
                          attn_mask=attn_mask, need_weights=False)[0]  # (seq_len, batch, embed_dim)

        if self.c_attn is not None:
            tgt_len = x.size(0)
            x = x.view(tgt_len, -1, self.num_heads, self.head_dim)
            x = torch.einsum('tbhd,h->tbdh', x, self.c_attn)
            x = x.reshape(tgt_len, -1, self.embed_dim)
        if self.post_attn_norm is not None:
            x = self.post_attn_norm(x)
        x = self.dropout(x)
        x += residual

        residual = x
        x = self.pre_fc_norm(x)
        x = self.act(self.fc1(x))
        x = self.act_dropout(x)
        if self.post_fc_norm is not None:
            x = self.post_fc_norm(x)
        x = self.fc2(x)
        x = self.dropout(x)
        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x += residual

        return x
    
## function and module to flip gradient                                                                                                                                                               
class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x,alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        _, alpha = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
             grad_input = - alpha*grad_output
        return grad_input, None

class GradientReverse(nn.Module):
    def __init__(self, alpha=1., *args, **kwargs):
        """                                                                                                                                                                                         
        A gradient reversal layer. This layer has no parameters, and simply reverses the gradient in the backward pass.                                                                             
        """
        super().__init__(*args, **kwargs)
        self.alpha = torch.tensor(alpha, requires_grad=False)
    def forward(self, x):
        return RevGrad.apply(x, self.alpha)

class ParticleTransformer(nn.Module):

    def __init__(self,
                 input_dim,
                 num_classes=None,
                 num_targets=None,
                 num_domains=[],
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 embed_dims=[128, 512, 128],
                 pair_embed_dims=[64, 64, 64],
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 # dense layers
                 fc_params=[],
                 fc_domain_params=[],
                 fc_contrastive_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 split_domain_outputs=False,
                 split_reg_outputs=False,
                 use_contrastive_domain=False,
                 alpha_grad=1,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.for_inference = for_inference
        self.use_amp = use_amp
        self.num_classes = num_classes;
        self.num_targets = num_targets;
        self.num_domains = num_domains;
        self.alpha_grad = alpha_grad;
        self.fc_domain = None;
        self.fc_contrastive = None;
        self.fc_contrastive_da = None;
        self.split_domain_outputs = split_domain_outputs;
        self.split_reg_outputs = split_reg_outputs;
        self.save_grad_inputs = False;
        self.use_contrastive_domain = use_contrastive_domain;
        
        embed_dim = embed_dims[-1] if len(embed_dims) > 0 else input_dim
        default_cfg = dict(embed_dim=embed_dim, num_heads=num_heads, ffn_ratio=4,
                           dropout=0.1, attn_dropout=0.1, activation_dropout=0.1,
                           add_bias_kv=False, activation=activation,
                           scale_fc=True, scale_attn=True, scale_heads=True, scale_resids=True)

        cfg_block = copy.deepcopy(default_cfg)
        if block_params is not None:
            cfg_block.update(block_params)

        cfg_cls_block = copy.deepcopy(default_cfg)
        if cls_block_params is not None:
            cfg_cls_block.update(cls_block_params)

        self.pair_extra_dim = pair_extra_dim

        ## embedding
        self.embed = Embed(input_dim, embed_dims, activation=activation) if len(embed_dims) > 0 else nn.Identity()
        self.pair_embed = PairEmbed(
            pair_input_dim, pair_extra_dim, pair_embed_dims + [cfg_block['num_heads']],
            remove_self_pair=remove_self_pair, use_pre_activation_pair=use_pre_activation_pair,
            for_onnx=for_inference) if pair_embed_dims is not None and pair_input_dim + pair_extra_dim > 0 else None
        ## transformer blocks
        self.blocks = nn.ModuleList([Block(**cfg_block) for _ in range(num_layers)])
        ## class tokens
        self.cls_blocks = nn.ModuleList([Block(**cfg_cls_block) for _ in range(num_cls_layers)])
        ## normalization
        self.norm = nn.LayerNorm(embed_dim)
                                                
        ## fully connected layers
        fcs = []
        fcs_reg = []
        if fc_params:
            in_dim = embed_dim
            for out_dim, drop_rate in fc_params:
                fcs.append(nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.GELU() if activation == 'gelu' else nn.ReLU(),
                    nn.Dropout(drop_rate))
                )
                if self.split_reg_outputs:
                    fcs_reg.append(nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                        nn.Dropout(drop_rate))
                    )                    
                in_dim = out_dim
            if self.split_reg_outputs:
                fcs.append(nn.Linear(in_dim, num_classes))
                fcs_reg.append(nn.Linear(in_dim, num_targets))
                self.fc = nn.Sequential(*fcs)
                self.fc_reg = nn.Sequential(*fcs_reg)
            else:
                fcs.append(nn.Linear(in_dim,num_classes+num_targets))
                self.fc = nn.Sequential(*fcs)
        else:
            self.fc = None
            self.fc_reg = None

        ## contrastive projection
        if fc_contrastive_params :
            in_dim = embed_dim
            fcs_contrastive = []
            for out_dim, drop_rate in fc_contrastive_params:
                 fcs_contrastive.append(nn.Sequential(
                     nn.Linear(in_dim, out_dim),
                     nn.BatchNorm1d(out_dim),
                     nn.GELU() if activation == 'gelu' else nn.ReLU(),
                     nn.Dropout(drop_rate))
                 )
                 in_dim = out_dim
            fcs_contrastive.append(nn.Linear(in_dim,in_dim));
            self.fc_contrastive = nn.Sequential(*fcs_contrastive);
        else:
            self.fc_contrastive = None;
            
        ## domain layers
        if not for_inference and self.num_domains:
            ## contrastive da projection
            fcs_contrastive_da = []
            if fc_contrastive_params and  use_contrastive_domain:
                in_dim = embed_dim
                fcs_contrastive_da.append(GradientReverse(self.alpha_grad));
                for out_dim, drop_rate in fc_contrastive_params:
                    fcs_contrastive_da.append(nn.Sequential(
                        nn.Linear(in_dim, out_dim),
                        nn.BatchNorm1d(out_dim),
                        nn.GELU() if activation == 'gelu' else nn.ReLU(),
                        nn.Dropout(drop_rate))
                    )
                    in_dim = out_dim
                fcs_contrastive_da.append(nn.Linear(in_dim,in_dim));
                self.fc_contrastive_da = nn.Sequential(*fcs_contrastive_da);
            else:
                self.fc_contrastive_da = None;
            ## standard domain layers
            if not self.split_domain_outputs:
                num_domain = sum(element for element in self.num_domains);
                fcs_domain = []
                fcs_domain.append(GradientReverse(self.alpha_grad));
                for idx, layer_param in enumerate(fc_domain_params):
                    channels, drop_rate = layer_param
                    if idx == 0:
                        in_chn = embed_dim
                    else:
                        in_chn = fc_domain_params[idx - 1][0]
                    fcs_domain.append(
                        nn.Sequential(
                            nn.Linear(in_chn, channels),
                            nn.BatchNorm1d(channels),
                            nn.GELU() if activation == 'gelu' else nn.ReLU(),
                            nn.Dropout(drop_rate)
                        )
                    )
                fcs_domain.append(nn.Linear(fc_domain_params[-1][0], num_domain))
                self.fc_domain = nn.Sequential(*fcs_domain)
            else:
                for idd,dom in enumerate(self.num_domains):
                    fcs_domain = [];
                    fcs_domain.append(GradientReverse(self.alpha_grad));
                    for idx, layer_param in enumerate(fc_domain_params):
                        channels, drop_rate = layer_param
                        if idx == 0:
                            in_chn = embed_dim
                        else:
                            in_chn = fc_domain_params[idx - 1][0]
                        
                        fcs_domain.append(nn.Sequential(
                            nn.Linear(in_chn, channels),
                            nn.BatchNorm1d(channels),
                            nn.GELU() if activation == 'gelu' else nn.ReLU(),
                            nn.Dropout(drop_rate))
                        )

                    fcs_domain.append(nn.Linear(fc_domain_params[-1][0],dom))
                    if self.fc_domain is None:
                        self.fc_domain = nn.ModuleList([nn.Sequential(*fcs_domain)]);
                    else:
                        self.fc_domain.append(nn.Sequential(*fcs_domain));

        # init
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token', }

    def forward(self, x, v=None, mask=None, uu=None, uu_idx=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        # for pytorch: uu (N, C', num_pairs), uu_idx (N, 2, num_pairs)
        # for onnx: uu (N, C', P, P), uu_idx=None
        with torch.set_grad_enabled(self.save_grad_inputs):
            if not self.for_inference:
                if uu_idx is not None:
                    uu = build_sparse_tensor(uu, uu_idx, x.size(-1))
            x, v, mask, uu = self.trimmer(x, v, mask, uu)
            padding_mask = ~mask.squeeze(1)  # (N, P)

        with torch.cuda.amp.autocast(enabled=self.use_amp):

            # input embedding
            x = self.embed(x).masked_fill(~mask.permute(2, 0, 1), 0)  # (P, N, C)
            attn_mask = None
            if (v is not None or uu is not None) and self.pair_embed is not None:
                self.pair_embed.save_grad_inputs = self.save_grad_inputs;
                attn_mask = self.pair_embed(v, uu).view(-1, v.size(-1), v.size(-1))  # (N*num_heads, P, P)

            # transform
            for block in self.blocks:
                x = block(x, x_cls=None, padding_mask=padding_mask, attn_mask=attn_mask)

            # extract class token
            cls_tokens = self.cls_token.expand(1, x.size(1), -1)  # (1, N, C)
            for block in self.cls_blocks:
                cls_tokens = block(x, x_cls=cls_tokens, padding_mask=padding_mask)

            x_cls = self.norm(cls_tokens).squeeze(0)
            
            ### if there are no fully connected output resturns x_cls
            if self.fc is None:
                return x_cls

            ### classification and regression output
            if self.split_reg_outputs:
                output = self.fc(x_cls)
                output_reg = self.fc_reg(x_cls)
            else:
                output = self.fc(x_cls)

            ### buld the final output to be returned to the main function
            if self.for_inference:
                if self.num_classes and not self.num_targets:
                    output = torch.softmax(output, dim=1)                    
                elif self.num_classes and self.num_targets:
                    if self.split_reg_outputs:
                        output_class = torch.softmax(output,dim=1);
                        output = torch.cat((output_class,output_reg),dim=1);
                    else:
                        output_class = torch.softmax(output[:,:self.num_classes],dim=1);
                        output_reg = output[:,self.num_classes:self.num_classes+self.num_targets];
                        output = torch.cat((output_class,output_reg),dim=1);
            elif self.num_domains and self.fc_domain:
                if not self.split_domain_outputs:
                    output_domain = self.fc_domain(x_cls)
                    if self.split_reg_outputs:
                        output = torch.cat((output,output_reg,output_domain),dim=1);
                    else:
                        output = torch.cat((output,output_domain),dim=1);
                else:
                    if self.split_reg_outputs:
                        output = torch.cat((output,output_reg),dim=1);
                    for i,fc in enumerate(self.fc_domain):
                        output_domain = fc(x_cls);
                        output = torch.cat((output,output_domain),dim=1);

            ### contrastive output
            if self.fc_contrastive is not None and self.fc_contrastive_da is not None:
                output_cont = self.fc_contrastive(x_cls);
                output_cont_da = self.fc_contrastive_da(x_cls);
                return output, output_cont, output_cont_da;
            elif self.fc_contrastive is not None:
                output_cont = self.fc_contrastive(x_cls);
                return output, output_cont;            
            else:
                return output

class ParticleTransformerTagger(nn.Module):

    def __init__(self,
                 # input tensors
                 pf_input_dim,
                 sv_input_dim,
                 lt_input_dim,
                 # output of network
                 num_classes=None,
                 num_targets=None,
                 num_domains=[],                 
                 # network configurations
                 pair_input_dim=4,
                 pair_extra_dim=0,
                 remove_self_pair=False,
                 use_pre_activation_pair=True,
                 # feature embeddings
                 embed_dims=[128, 512, 128],
                 # pair embeddings
                 pair_embed_dims=[64, 64, 64],
                 # layers layout
                 num_heads=8,
                 num_layers=8,
                 num_cls_layers=2,
                 block_params=None,
                 cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
                 # final fully connected layers
                 fc_params=[],
                 fc_domain_params=[],
                 fc_contrastive_params=[],
                 activation='gelu',
                 # misc
                 trim=True,
                 for_inference=False,
                 use_amp=False,
                 # options for splitting domain and regression outputs
                 split_domain_outputs=False,
                 split_reg_outputs=False,
                 use_contrastive_domain=False,
                 # save gradiantes for fgsm
                 save_grad_inputs=False,
                 alpha_grad=1,
                 **kwargs) -> None:

        super().__init__(**kwargs)

        self.use_amp = use_amp
        self.save_grad_inputs = False if for_inference else save_grad_inputs;
        
        self.pf_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.sv_trimmer = SequenceTrimmer(enabled=trim and not for_inference)
        self.lt_trimmer = SequenceTrimmer(enabled=trim and not for_inference)        
        self.pf_embed = Embed(pf_input_dim, embed_dims, activation=activation)
        self.sv_embed = Embed(sv_input_dim, embed_dims, activation=activation)
        self.lt_embed = Embed(lt_input_dim, embed_dims, activation=activation)
        
        self.part = ParticleTransformer(
            input_dim=embed_dims[-1],
            num_classes=num_classes,
            num_targets=num_targets,
            num_domains=num_domains,
            ## network configurations
            pair_input_dim=pair_input_dim,
            pair_extra_dim=pair_extra_dim,
            remove_self_pair=remove_self_pair,
            use_pre_activation_pair=use_pre_activation_pair,
            ## transformer blocks
            embed_dims=[],
            pair_embed_dims=pair_embed_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            num_cls_layers=num_cls_layers,
            block_params=block_params,
            cls_block_params=cls_block_params,
            ## dense layers
            fc_params=fc_params,
            fc_domain_params=fc_domain_params,
            fc_contrastive_params=fc_contrastive_params,
            activation=activation,
            ## misc
            trim=False,
            for_inference=for_inference,
            use_amp=self.use_amp,
            ## domain and contrastive
            split_domain_outputs=split_domain_outputs,
            split_reg_outputs=split_reg_outputs,
            use_contrastive_domain=use_contrastive_domain,
            alpha_grad=alpha_grad
        )

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'part.cls_token', }

    def forward(self, pf_x, pf_v=None, pf_mask=None, sv_x=None, sv_v=None, sv_mask=None, lt_x=None, lt_v=None, lt_mask=None):
        # x: (N, C, P)
        # v: (N, 4, P) [px,py,pz,energy]
        # mask: (N, 1, P) -- real particle = 1, padded = 0
        with torch.set_grad_enabled(self.save_grad_inputs):
            pf_x, pf_v, pf_mask, _ = self.pf_trimmer(pf_x, pf_v, pf_mask)
            sv_x, sv_v, sv_mask, _ = self.sv_trimmer(sv_x, sv_v, sv_mask)
            lt_x, lt_v, lt_mask, _ = self.lt_trimmer(lt_x, lt_v, lt_mask)
            v    = torch.cat([pf_v, sv_v, lt_v], dim=2)
            mask = torch.cat([pf_mask, sv_mask, lt_mask], dim=2)
            
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            pf_x = self.pf_embed(pf_x)  # after embed: (seq_len, batch, embed_dim)
            sv_x = self.sv_embed(sv_x)
            lt_x = self.lt_embed(lt_x)
            x = torch.cat([pf_x, sv_x, lt_x], dim=0)
            self.part.save_grad_inputs = self.save_grad_inputs
            return self.part(x, v, mask)


