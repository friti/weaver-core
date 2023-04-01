import numpy as np
import torch
import torch.nn as nn

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2).contiguous()  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0).contiguous()  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)

## function and module to flip gradient
class RevGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x,alpha)
        return x

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
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
        
class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 num_targets,
                 num_domains=0,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 fc_domain_params=[],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 use_revgrad=True,
                 use_domain_on_output=False,
                 for_inference=False,
                 alpha_grad=1,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.num_classes = num_classes;
        self.num_targets = num_targets;
        self.num_domains = num_domains;
        self.alpha_grad = alpha_grad;
        self.use_fts_bn = use_fts_bn
        self.for_inference = for_inference
        self.use_counts = use_counts        
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)
        self.fc_domain = None;
        self.use_domain_on_output = use_domain_on_output;

        # Edge Conv blocks
        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion

        if self.use_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        # fully connected layers for classification
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]

            fcs.append(nn.Sequential(
                nn.Linear(in_chn, channels), 
                nn.ReLU(), 
                nn.Dropout(drop_rate)))
            if self.use_domain_on_output:
                out_chan_fcs = channels
        if self.use_domain_on_output:
            self.fc_minus_one = nn.Sequential(*fcs)
        fcs.append(nn.Linear(fc_params[-1][0], num_classes+num_targets))
        self.fc = nn.Sequential(*fcs)
                
        # add or not the domain layers
        fcs_domain = []
        if not num_domains:
            if not for_inference:
                if use_revgrad:
                    fcs_domain.append(GradientReverse(self.alpha_grad));
                for idx, layer_param in enumerate(fc_domain_params):
                    channels, drop_rate = layer_param
                    if idx == 0:
                        if self.use_domain_on_output:
                            in_chn = out_chan_fcs
                        else:
                            in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
                    else:
                        in_chn = fc_domain_params[idx - 1][0]

                    fcs_domain.append(nn.Sequential(
                        nn.Linear(in_chn, channels), 
                        nn.ReLU(), 
                        nn.Dropout(drop_rate)))

                fcs_domain.append(nn.Linear(fc_domain_params[-1][0], num_domains))             
                self.fc_domain = nn.Sequential(*fcs_domain)

    def forward(self, points, features, mask=None):

        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)

        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9

        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features

        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)

        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

        if self.use_counts:
            x = fts.sum(dim=-1) / counts  # divide by the real counts
        else:
            x = fts.mean(dim=-1)
        
        output = self.fc(x)

        if self.for_inference:
            if self.num_classes and not self.num_targets:
                output = torch.softmax(output,dim=1);
            elif self.num_classes and self.num_targets:
                output_class = torch.softmax(output[:,:self.num_classes],dim=1)
                output_reg   = output[:,self.num_classes:self.num_classes+self.num_targets];
                output = torch.cat((output_class,output_reg),dim=1);
        elif self.num_domains and self.fc_domain:
            if self.use_domain_on_output:
                output_domain = self.fc_domain(self.fc_minus_one(x))
            else:
                output_domain = self.fc_domain(x)
            output = torch.cat((output,output_domain),dim=1);
                    
        return output

class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 num_targets,
                 num_domains=0,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 fc_domain_params=[],
                 input_dims=32,
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 use_revgrad=True,
                 use_domain_on_output=False,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 alpha_grad=1,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, input_dims)
        self.sv_conv = FeatureConv(sv_features_dims, input_dims)
        self.pn = ParticleNet(input_dims=input_dims,
                              num_classes=num_classes,
                              num_targets=num_targets,
                              num_domains=num_domains,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              fc_domain_params=fc_domain_params,
                              use_revgrad=use_revgrad,
                              use_domain_on_output=use_domain_on_output,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference,
                              alpha_grad=alpha_grad
        )

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=2)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        return self.pn(points, features, mask)


class ParticleNetLostTrkTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 lt_features_dims,
                 num_classes,
                 num_targets,
                 num_domains=0,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 fc_domain_params=[],
                 input_dims=32,
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 use_revgrad=True,
                 use_domain_on_output=False,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 lt_input_dropout=None,
                 for_inference=False,
                 alpha_grad=1,
                 **kwargs):
        super(ParticleNetLostTrkTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.lt_input_dropout = nn.Dropout(lt_input_dropout) if lt_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, input_dims)
        self.sv_conv = FeatureConv(sv_features_dims, input_dims)
        self.lt_conv = FeatureConv(lt_features_dims, input_dims)
        self.pn = ParticleNet(input_dims=input_dims,
                              num_classes=num_classes,
                              num_targets=num_targets,
                              num_domains=num_domains,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              fc_domain_params=fc_domain_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              use_revgrad=use_revgrad,
                              use_domain_on_output=use_domain_on_output,
                              for_inference=for_inference,
                              alpha_grad=alpha_grad)

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask, lt_points, lt_features, lt_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask
        if self.lt_input_dropout:
            lt_mask = (self.lt_input_dropout(lt_mask) != 0).float()
            lt_points *= lt_mask
            lt_features *= lt_mask

        points = torch.cat((pf_points, sv_points, lt_points), dim=2)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask, self.lt_conv(lt_features*lt_mask)*lt_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask, lt_mask), dim=2)
        return self.pn(points, features, mask)
