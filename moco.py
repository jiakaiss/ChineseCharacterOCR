import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Net


def _build_mlp(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim
        mlp.append(nn.Linear(dim1, dim2, bias=False))
        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR 's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))
    return nn.Sequential(*mlp)


class HW_MoCo(nn.Module):
    def __init__(self, dim=128, mlp_dim=2048, T=0.2, fea_dim=512, pre_path=None):
        super(HW_MoCo, self).__init__()
        self.T = T

        # build encoders
        self.base_encoder, self.momentum_encoder = Net().resnet, Net().resnet

        self.base_projector = _build_mlp(2, fea_dim, mlp_dim, dim)
        self.momentum_projector = _build_mlp(2, fea_dim, mlp_dim, dim)

        self.predictor = _build_mlp(2, dim, mlp_dim, dim, False)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        if pre_path is not None:
            print('load pretrained model...')
            self.load_state_dict(torch.load(pre_path, map_location='cpu'), strict=True)

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

        for param_b, param_m in zip(self.base_projector.parameters(), self.momentum_projector.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q, k = F.normalize(q, dim=1), F.normalize(k, dim=1)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """
        # compute features
        q1 = self.predictor(self.base_projector(self.base_encoder(x1)))
        q2 = self.predictor(self.base_projector(self.base_encoder(x2)))

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder
            # compute momentum features as targets
            k1 = self.momentum_projector(self.momentum_encoder(x1))
            k2 = self.momentum_projector(self.momentum_encoder(x2))
        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

