import torch
from torch import nn

class ODEFunc(nn.Module):
    # inspired by https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
    # MLP with time dependancy
    # Probably not the most optimal one !!

    # TODO : Possible improvement :
    # - add a time embedding
    # - concatenate time at each layer

    def __init__(self, dims, bn=False):
        super(ODEFunc, self).__init__()

        self.dims = dims
        self.bn = bn

        layers = [nn.Linear(dims[0] + 1, dims[1])]
        for i in range(len(dims) - 2):
            layers.append(nn.ReLU())
            if self.bn:
                layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.Linear(dims[i+1], dims[i+2]))

        self.net = nn.Sequential(*layers)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y, t):
        # y : shape (n, d)
        # t : shape (n,)
        return self.net(torch.cat([y, t.unsqueeze(-1)], dim=-1))