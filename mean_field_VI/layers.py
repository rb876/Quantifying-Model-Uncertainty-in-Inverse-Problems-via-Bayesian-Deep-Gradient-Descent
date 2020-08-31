import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layer_utils as utils

class LinearVariance(nn.Linear):
    def __init__(self, in_features, out_features, bias):
        super().__init__(in_features, out_features, bias)
        self.softplus = nn.Softplus()

    @property
    def w_var(self):
        return 1e-6 + self.softplus(self.weight) ** 2

    def forward(self, x):
        return torch.nn.functional.linear(x ** 2, self.w_var, bias=self.bias)

class LocalReparamDense(nn.Module):
    '''
    A wrapper module for functional dense layer that performs local reparametrization
    '''
    def __init__(self, shape):
        super().__init__()
        self.in_features, self.out_features = shape
        self.mean = nn.Linear(
            in_features=self.in_features,
            out_features=self.out_features,
            bias=True
        )

        self.var = LinearVariance(self.in_features, self.out_features, bias=False)
        nn.init.normal_(self.mean.weight, 0, 0.005)
        nn.init.normal_(self.var.weight, -9., 0.005)

    def forward(self, x, num_samples=1, squeeze=False):
        mean, var = self.mean(x), self.var(x)
        return utils.sample_normal(mean, var, num_samples, squeeze)

    def compute_kl(self):
        mean, cov = self._compute_posterior()
        # scale = 2. / self.mean.weight.shape[0]
        scale = 1.
        return utils.gaussian_kl_diag(mean, cov, torch.zeros_like(mean), scale * torch.ones_like(mean))

    def _compute_posterior(self):
        return self.mean.weight.flatten(), self.var.w_var.flatten()

class LocalReparamConv2d(LocalReparamDense):
    '''
    A wrapper module for conv2d that performs local reparametrization
    '''
    def __init__(self, shape, hyperprior=None, initial_logstd=-8.):
        super(LocalReparamDense, self).__init__()
        in_channels, out_channels, kernel_size, stride = shape
        self.mean = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=True
        )
        self.var = Conv2dVariance(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            bias=False
        )

        nn.init.normal_(self.mean.weight, 0., 0.05)
        # nn.init.uniform_(self.mean.weight, 0.01, 0.001)
        nn.init.normal_(self.var.weight, -9., 0.05)

class Conv2dVariance(nn.Conv2d):
    def __init__(self,  in_channels, out_channels, kernel_size=3, bias=False, stride=1, padding=1):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, bias=bias, stride=stride, padding=padding)

    @property
    def w_var(self):
        return torch.exp(self.weight) ** 2

    def forward(self, x):
        return F.conv2d(x ** 2, self.w_var, self.bias, self.stride, self.padding)
