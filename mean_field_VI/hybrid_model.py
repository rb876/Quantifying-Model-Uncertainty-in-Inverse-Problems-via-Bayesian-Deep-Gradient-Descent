import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import layer_utils as utils

from neural_trainer_mfvi import NeuralTrainer
from layers import LocalReparamConv2d

class HybridModel(NeuralTrainer):
    def __init__(self, arch_args):
        super().__init__()
        self.det_upper_layers, self.det_lower_layers, self.det_common_layers = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.log_noise = nn.Parameter(torch.cuda.FloatTensor([0]))

        for shape in arch_args['arch']['up']:
            layer = torch.nn.Conv2d
            self.det_upper_layers.append(layer(*shape))

        for shape in arch_args['arch']['low']:
            layer = torch.nn.Conv2d
            self.det_lower_layers.append(layer(*shape))

        for shape in arch_args['arch']['cm']:
            layer = torch.nn.Conv2d
            self.det_common_layers.append(layer(*shape))

        self.BayesCNN = LocalReparamConv2d([16, 1, 3, 1])

    def forward(self, x, grad, num_samples=1):
        skip = x
        x = self.encode(x, grad)
        x = self.BayesCNN(x, num_samples=num_samples, squeeze=True)
        return F.relu(x + skip)

    def encode(self, x, grad):
        for layer in self.det_upper_layers:
            x = layer(x)
            x = F.relu(x)
        for layer in self.det_lower_layers:
            grad = layer(grad)
            grad = F.relu(grad)
        x = torch.cat((x, grad), dim=1)
        for layer in self.det_common_layers:
            x = layer(x)
            x = F.relu(x)
        return x

    def _compute_loss(self, x, x_pred, batch_size, data_size):
        # The objective is 1/n * (\sum_i log_like_i - KL)
        kl = utils.to_gpu(torch.tensor([0.]))
        log_likelihood = self._compute_log_likelihood(x, x_pred, self.log_noise.exp()**2)
        kl = self._compute_kl() * (batch_size / data_size)
        elbo = log_likelihood - kl
        return -elbo, kl

    def _compute_kl(self):
        return self.BayesCNN.compute_kl()
