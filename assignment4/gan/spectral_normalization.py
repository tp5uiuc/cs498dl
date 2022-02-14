import torch
from torch.optim.optimizer import Optimizer, required

from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        """
        Reference:
        SPECTRAL NORMALIZATION FOR GENERATIVE ADVERSARIAL NETWORKS: https://arxiv.org/pdf/1802.05957.pdf
        """
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        """
        TODO: Implement Spectral Normalization
        Hint: 1: Use getattr to first extract u, v, w.
              2: Apply power iteration.
              3: Calculate w with the spectral norm.
              4: Use setattr to update w in the module.
        """

        # w still gets gradient from god knows where
        w = getattr(self.module, self.name + "_bar")
        # print("W_Bar:" ,w.view(w.data.shape[0], -1)[0, :10])
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")

        # we don't need autograd here, safety
        with torch.no_grad():
            # Do power iterations
            # https://jonathan-hui.medium.com/gan-spectral-normalization-893b6a4e8f53
            for _ in range(self.power_iterations):
                # w is of size (batch_size, n_features, k, k)
                # so we need to reshape it as (batch_size, n_features * k * k) first
                # https://github.com/pfnet-research/sngan_projection/blob/e84b1a5f604de5fec268f37c3f26478e80b7f475/source/links/sn_convolution_2d.py#L74
                reshaped_w = w.view(w.data.shape[0], -1)  # from line 69
                w_t = torch.transpose(reshaped_w, 0, 1)
                v.data = l2normalize(torch.matmul(w_t.data, u.data))
                cache = torch.matmul(reshaped_w.data, v.data)
                u.data = l2normalize(cache)

        sigma = torch.dot(u.data, cache)
        # print("sigma", sigma)

        # set to w here so that Conv2D knows the weights
        setattr(self.module, self.name, w / sigma)
        # print("after W_Bar:" ,w.view(w.data.shape[0], -1)[0, :10])

    def _make_params(self):
        """
        No need to change. Initialize parameters.
        v: Initialize v with a random vector (sampled from isotropic distrition).
        u: Initialize u with a random vector (sampled from isotropic distrition).
        w: Weight of the current layer.
        """
        w = getattr(self.module, self.name)

        # (out_channels,  groups, in_channels, \text{kernel\_size[0]}, \text{kernel\_size[1]})
        # height = out-channels
        height = w.data.shape[0]
        # width = in_channels * kernel\_size[0] * kernel\_size[1]
        width = w.view(height, -1).data.shape[1]
        # print("Height", height)
        # print("Width", width)

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        """
        No need to change. Update weights using spectral normalization.
        """
        self._update_u_v()
        return self.module.forward(*args)
