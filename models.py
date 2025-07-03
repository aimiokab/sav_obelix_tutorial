import torch
from torch import nn

class cnf(torch.nn.Module):
    """Transform the model used by the FLOW_MATCHING class into a function with 
    a signature adapted to odeint.

    Coming from: https://github.com/annegnx/PnP-Flow/
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t:torch.Tensor, x:torch.Tensor):
        with torch.no_grad():
            # z = self.model(x, t.squeeze())
            z = self.model(x, t.repeat(x.shape[0]))
        return z

class ODEFunc(nn.Module):
    """
    inspired by https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py
    
    MLP with time dependancy (concatenated to the input).
    
    Probably not the most optimal one !!
    """

    # TODO : Possible improvements :
    # - add a time embedding
    # - concatenate time at each layer

    def __init__(self, dims, bn=False):
        """
        Args:
            dims (tuple[int]): successive dimensions (input_dim, *hidden_dims, output_dim). 
                Time must not be counted in input_dim. 
            bn (bool, optional): whether to use batch normalisation. Defaults to False.
        """
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
        """
        Args:
            y (Tensor): shape (N, input_dim)
            t (Tensor): shape (N,)

        Returns:
            Tensor: shape (N, output_dim)
        """
        return self.net(torch.cat([y, t.unsqueeze(-1)], dim=-1))